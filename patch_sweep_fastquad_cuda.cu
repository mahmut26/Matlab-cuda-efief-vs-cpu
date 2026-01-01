// patch_sweep_fastquad_cuda.cu
#include "mex.h"
#include "matrix.h"

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cusolverDn.h>
#include <cublas_v2.h>

#include <math.h>
#include <stdint.h>

#ifndef PI_D
#define PI_D 3.141592653589793238462643383279502884
#endif

#define CUDA_OK(call) do{ cudaError_t e=(call); if(e!=cudaSuccess){ mexErrMsgIdAndTxt("cuda:rt", cudaGetErrorString(e)); } }while(0)
#define CUSOLVER_OK(call) do{ cusolverStatus_t s=(call); if(s!=CUSOLVER_STATUS_SUCCESS){ mexErrMsgIdAndTxt("cuda:cusolver","cuSOLVER error"); } }while(0)
#define CUBLAS_OK(call) do{ cublasStatus_t s=(call); if(s!=CUBLAS_STATUS_SUCCESS){ mexErrMsgIdAndTxt("cuda:cublas","cuBLAS error"); } }while(0)

static inline void check(bool cond, const char* id, const char* msg){
    if(!cond) mexErrMsgIdAndTxt(id, msg);
}

static inline cuDoubleComplex get_complex_scalar(const mxArray* a){
    double re = mxGetScalar(a);
    double im = 0.0;
    if(mxIsComplex(a)){
#if MX_HAS_INTERLEAVED_COMPLEX
        const mxComplexDouble* z = mxGetComplexDoubles(a);
        re = z[0].real; im = z[0].imag;
#else
        double* pi = mxGetPi(a);
        if(pi) im = pi[0];
#endif
    }
    return make_cuDoubleComplex(re, im);
}

struct d3 { double x,y,z; };

__host__ __device__ static inline d3 make_d3(double x,double y,double z){ d3 a{x,y,z}; return a; }
__host__ __device__ static inline d3 sub3(d3 a,d3 b){ return make_d3(a.x-b.x,a.y-b.y,a.z-b.z); }
__host__ __device__ static inline double dot3(d3 a,d3 b){ return a.x*b.x + a.y*b.y + a.z*b.z; }
__host__ __device__ static inline double norm3(d3 a){ return sqrt(dot3(a,a)); }

__device__ static inline cuDoubleComplex cadd(cuDoubleComplex a, cuDoubleComplex b){
    return make_cuDoubleComplex(cuCreal(a)+cuCreal(b), cuCimag(a)+cuCimag(b));
}
__device__ static inline cuDoubleComplex cmul(cuDoubleComplex a, cuDoubleComplex b){
    double ar=cuCreal(a), ai=cuCimag(a);
    double br=cuCreal(b), bi=cuCimag(b);
    return make_cuDoubleComplex(ar*br - ai*bi, ar*bi + ai*br);
}
__device__ static inline cuDoubleComplex cmul_real(cuDoubleComplex a, double s){
    return make_cuDoubleComplex(cuCreal(a)*s, cuCimag(a)*s);
}

// exp(-j*(kR)) where k = kr + j*ki => exp(ki R) * (cos(krR) - j sin(krR))
__device__ static inline cuDoubleComplex exp_negj_kR(cuDoubleComplex k, double R){
    double kr = cuCreal(k);
    double ki = cuCimag(k);
    double amp = exp(ki * R);
    double ph  = kr * R;
    return make_cuDoubleComplex(amp*cos(ph), -amp*sin(ph));
}
__device__ static inline cuDoubleComplex green(cuDoubleComplex k, double R){
    double inv = 1.0 / (4.0 * PI_D * R);
    cuDoubleComplex e = exp_negj_kR(k, R);
    return cmul_real(e, inv);
}

// RWG eval at point r on triangle t (t is 1-based)
__device__ static inline void rwg_eval_point(
    int t, int tp, int tm,
    int plusSign, int minusSign,
    double len, double Ap, double Am,
    d3 rp, d3 rm,
    d3 r,
    d3 &f_out, double &div_out
){
    if(t == tp){
        double s = (double)plusSign;
        double scale = s * (len/(2.0*Ap));
        f_out = make_d3(scale*(r.x-rp.x), scale*(r.y-rp.y), scale*(r.z-rp.z));
        div_out = s * (len/Ap);
        return;
    }
    if(t == tm){
        double s = (double)minusSign;
        double scale = s * (len/(2.0*Am));
        f_out = make_d3(scale*(rm.x-r.x), scale*(rm.y-r.y), scale*(rm.z-r.z));
        div_out = s * (-(len/Am));
        return;
    }
    f_out = make_d3(0,0,0);
    div_out = 0.0;
}

// regular quad (cached)
__device__ static inline void integral_regular(
    const double* r_reg, const double* w_reg,
    int Nq, int Nt,
    int m, int n,
    int t1, int t2,
    const int* plusTri, const int* minusTri,
    const int* plusSign, const int* minusSign,
    const d3* rp, const d3* rm,
    const double* Ap, const double* Am,
    const double* len,
    cuDoubleComplex k,
    cuDoubleComplex &Ivec, cuDoubleComplex &Isca
){
    int t1z = t1-1;
    int t2z = t2-1;

    int tp_m = plusTri[m], tm_m = minusTri[m];
    int tp_n = plusTri[n], tm_n = minusTri[n];

    d3 rp_m = rp[m], rm_m = rm[m];
    d3 rp_n = rp[n], rm_n = rm[n];

    double lm = len[m], ln = len[n];
    double Apm = Ap[m], Amm = Am[m];
    double Apn = Ap[n], Amn = Am[n];

    int sppm = plusSign[m], smpm = minusSign[m];
    int sppn = plusSign[n], smpn = minusSign[n];

    cuDoubleComplex Ivec_sum = make_cuDoubleComplex(0,0);
    cuDoubleComplex Isca_sum = make_cuDoubleComplex(0,0);

    for(int i=0;i<Nq;i++){
        double wi = w_reg[i + Nq*t1z];
        d3 ri = make_d3(
            r_reg[i + Nq*0 + Nq*3*t1z],
            r_reg[i + Nq*1 + Nq*3*t1z],
            r_reg[i + Nq*2 + Nq*3*t1z]
        );

        d3 fm; double divm;
        rwg_eval_point(t1, tp_m, tm_m, sppm, smpm, lm, Apm, Amm, rp_m, rm_m, ri, fm, divm);

        for(int j=0;j<Nq;j++){
            double wj = w_reg[j + Nq*t2z];
            double w  = wi * wj;

            d3 rj = make_d3(
                r_reg[j + Nq*0 + Nq*3*t2z],
                r_reg[j + Nq*1 + Nq*3*t2z],
                r_reg[j + Nq*2 + Nq*3*t2z]
            );

            d3 fn; double divn;
            rwg_eval_point(t2, tp_n, tm_n, sppn, smpn, ln, Apn, Amn, rp_n, rm_n, rj, fn, divn);

            d3 d = sub3(ri, rj);
            double R = norm3(d);
            if(R < 1e-15) R = 1e-15;

            cuDoubleComplex G = green(k, R);
            double dotff  = dot3(fm, fn);
            double divdiv = divm * divn;

            Ivec_sum = cadd(Ivec_sum, cmul_real(G, w*dotff));
            Isca_sum = cadd(Isca_sum, cmul_real(G, w*divdiv));
        }
    }

    Ivec = Ivec_sum;
    Isca = Isca_sum;
}

// near cached
__device__ static inline void integral_near(
    const double* near_r, const double* near_w,
    const double* aeqTri, double near_alpha,
    int Nq2, int NsN, int Nt,
    int m, int n,
    int t1, int t2,
    const int* plusTri, const int* minusTri,
    const int* plusSign, const int* minusSign,
    const d3* rp, const d3* rm,
    const double* Ap, const double* Am,
    const double* len,
    cuDoubleComplex k,
    cuDoubleComplex &Ivec, cuDoubleComplex &Isca
){
    int t1z = t1-1;
    int t2z = t2-1;

    double aeq = (aeqTri[t1z] < aeqTri[t2z]) ? aeqTri[t1z] : aeqTri[t2z];
    double a_soft = near_alpha * aeq;

    int tp_m = plusTri[m], tm_m = minusTri[m];
    int tp_n = plusTri[n], tm_n = minusTri[n];

    d3 rp_m = rp[m], rm_m = rm[m];
    d3 rp_n = rp[n], rm_n = rm[n];

    double lm = len[m], ln = len[n];
    double Apm = Ap[m], Amm = Am[m];
    double Apn = Ap[n], Amn = Am[n];

    int sppm = plusSign[m], smpm = minusSign[m];
    int sppn = plusSign[n], smpn = minusSign[n];

    cuDoubleComplex Ivec_sum = make_cuDoubleComplex(0,0);
    cuDoubleComplex Isca_sum = make_cuDoubleComplex(0,0);

    // near_r(i,xyz,p,t) => i + Nq2*xyz + Nq2*3*p + Nq2*3*NsN*t
    // near_w(i,p,t)     => i + Nq2*p + Nq2*NsN*t
    for(int p=0;p<NsN;p++){
        for(int i=0;i<Nq2;i++){
            double wi = near_w[i + Nq2*p + Nq2*NsN*t1z];
            d3 ri = make_d3(
                near_r[i + Nq2*0 + Nq2*3*p + Nq2*3*NsN*t1z],
                near_r[i + Nq2*1 + Nq2*3*p + Nq2*3*NsN*t1z],
                near_r[i + Nq2*2 + Nq2*3*p + Nq2*3*NsN*t1z]
            );

            d3 fm; double divm;
            rwg_eval_point(t1, tp_m, tm_m, sppm, smpm, lm, Apm, Amm, rp_m, rm_m, ri, fm, divm);

            for(int q=0;q<NsN;q++){
                for(int j=0;j<Nq2;j++){
                    double wj = near_w[j + Nq2*q + Nq2*NsN*t2z];
                    double w  = wi * wj;

                    d3 rj = make_d3(
                        near_r[j + Nq2*0 + Nq2*3*q + Nq2*3*NsN*t2z],
                        near_r[j + Nq2*1 + Nq2*3*q + Nq2*3*NsN*t2z],
                        near_r[j + Nq2*2 + Nq2*3*q + Nq2*3*NsN*t2z]
                    );

                    d3 fn; double divn;
                    rwg_eval_point(t2, tp_n, tm_n, sppn, smpn, ln, Apn, Amn, rp_n, rm_n, rj, fn, divn);

                    d3 d = sub3(ri, rj);
                    double R = norm3(d);
                    R = sqrt(R*R + a_soft*a_soft);
                    if(R < 1e-15) R = 1e-15;

                    cuDoubleComplex G = green(k, R);
                    double dotff  = dot3(fm, fn);
                    double divdiv = divm * divn;

                    Ivec_sum = cadd(Ivec_sum, cmul_real(G, w*dotff));
                    Isca_sum = cadd(Isca_sum, cmul_real(G, w*divdiv));
                }
            }
        }
    }

    Ivec = Ivec_sum;
    Isca = Isca_sum;
}

// self cached
__device__ static inline void integral_self(
    const double* self_r, const double* self_w, const double* self_a,
    int Nq2, int NsS, int Nt,
    int m, int n,
    int t,
    const int* plusTri, const int* minusTri,
    const int* plusSign, const int* minusSign,
    const d3* rp, const d3* rm,
    const double* Ap, const double* Am,
    const double* len,
    cuDoubleComplex k,
    cuDoubleComplex &Ivec, cuDoubleComplex &Isca
){
    int tz = t-1;

    int tp_m = plusTri[m], tm_m = minusTri[m];
    int tp_n = plusTri[n], tm_n = minusTri[n];

    d3 rp_m = rp[m], rm_m = rm[m];
    d3 rp_n = rp[n], rm_n = rm[n];

    double lm = len[m], ln = len[n];
    double Apm = Ap[m], Amm = Am[m];
    double Apn = Ap[n], Amn = Am[n];

    int sppm = plusSign[m], smpm = minusSign[m];
    int sppn = plusSign[n], smpn = minusSign[n];

    cuDoubleComplex Ivec_sum = make_cuDoubleComplex(0,0);
    cuDoubleComplex Isca_sum = make_cuDoubleComplex(0,0);

    // self_r(i,xyz,p,t) => i + Nq2*xyz + Nq2*3*p + Nq2*3*NsS*t
    // self_w(i,p,t)     => i + Nq2*p + Nq2*NsS*t
    // self_a(p,t)       => p + NsS*t
    for(int p=0;p<NsS;p++){
        double aS = self_a[p + NsS*tz];

        for(int i=0;i<Nq2;i++){
            double wi = self_w[i + Nq2*p + Nq2*NsS*tz];

            d3 ri = make_d3(
                self_r[i + Nq2*0 + Nq2*3*p + Nq2*3*NsS*tz],
                self_r[i + Nq2*1 + Nq2*3*p + Nq2*3*NsS*tz],
                self_r[i + Nq2*2 + Nq2*3*p + Nq2*3*NsS*tz]
            );

            d3 fm; double divm;
            rwg_eval_point(t, tp_m, tm_m, sppm, smpm, lm, Apm, Amm, rp_m, rm_m, ri, fm, divm);

            for(int q=0;q<NsS;q++){
                for(int j=0;j<Nq2;j++){
                    double wj = self_w[j + Nq2*q + Nq2*NsS*tz];
                    double w  = wi * wj;

                    d3 rj = make_d3(
                        self_r[j + Nq2*0 + Nq2*3*q + Nq2*3*NsS*tz],
                        self_r[j + Nq2*1 + Nq2*3*q + Nq2*3*NsS*tz],
                        self_r[j + Nq2*2 + Nq2*3*q + Nq2*3*NsS*tz]
                    );

                    d3 fn; double divn;
                    rwg_eval_point(t, tp_n, tm_n, sppn, smpn, ln, Apn, Amn, rp_n, rm_n, rj, fn, divn);

                    d3 d = sub3(ri, rj);
                    double R = norm3(d);

                    if(p==q){
                        R = sqrt(R*R + aS*aS);
                    }else{
                        if(R < 1e-15) R = 1e-15;
                    }

                    cuDoubleComplex G = green(k, R);
                    double dotff  = dot3(fm, fn);
                    double divdiv = divm * divn;

                    Ivec_sum = cadd(Ivec_sum, cmul_real(G, w*dotff));
                    Isca_sum = cadd(Isca_sum, cmul_real(G, w*divdiv));
                }
            }
        }
    }

    Ivec = Ivec_sum;
    Isca = Isca_sum;
}

__global__ void kernel_Z_fastquad(
    cuDoubleComplex* Z, int Ne,
    const int* plusTri, const int* minusTri,
    const int* plusSign, const int* minusSign,
    const d3* rp, const d3* rm,
    const double* Ap, const double* Am,
    const double* len,
    // pre
    const double* r_reg, const double* w_reg, int Nq, int Nt,
    const uint8_t* nearMat,
    const double* self_r, const double* self_w, const double* self_a, int Nq2, int NsS,
    const double* near_r, const double* near_w, int NsN,
    const double* aeqTri, double near_alpha,
    // physics
    cuDoubleComplex k,
    cuDoubleComplex alpha, cuDoubleComplex beta
){
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if(m>=Ne || n>=Ne) return;

    int tm[2] = { plusTri[m], minusTri[m] };
    int tn[2] = { plusTri[n], minusTri[n] };

    cuDoubleComplex Zmn = make_cuDoubleComplex(0,0);

    for(int im=0; im<2; ++im){
        int t1 = tm[im];
        if(t1==0) continue;

        for(int in=0; in<2; ++in){
            int t2 = tn[in];
            if(t2==0) continue;

            cuDoubleComplex Ivec, Isca;

            if(t1 == t2){
                integral_self(self_r, self_w, self_a, Nq2, NsS, Nt,
                              m, n, t1,
                              plusTri, minusTri, plusSign, minusSign,
                              rp, rm, Ap, Am, len,
                              k, Ivec, Isca);
            }else{
                int idx = (t1-1) + Nt*(t2-1); // column-major Nt x Nt
                if(nearMat[idx]){
                    integral_near(near_r, near_w, aeqTri, near_alpha,
                                  Nq2, NsN, Nt,
                                  m, n, t1, t2,
                                  plusTri, minusTri, plusSign, minusSign,
                                  rp, rm, Ap, Am, len,
                                  k, Ivec, Isca);
                }else{
                    integral_regular(r_reg, w_reg, Nq, Nt,
                                     m, n, t1, t2,
                                     plusTri, minusTri, plusSign, minusSign,
                                     rp, rm, Ap, Am, len,
                                     k, Ivec, Isca);
                }
            }

            Zmn = cadd(Zmn, cadd(cmul(alpha, Ivec), cmul(beta, Isca)));
        }
    }

    Z[m + n*Ne] = Zmn;
}

// host gaussian weights (CPU)
static void build_weights_gaussian(
    const double* V, mwSize Nv,
    const int* E, mwSize Ne,
    int edge0_1based, double spread,
    double* w_out, int* layer_out
){
    d3* c = (d3*)mxMalloc(Ne*sizeof(d3));
    double zmin=1e100, zmax=-1e100;

    for(mwSize e=0;e<Ne;++e){
        int a = E[e + 0*Ne] - 1;
        int b = E[e + 1*Ne] - 1;
        d3 ra = make_d3(V[a + 0*Nv], V[a + 1*Nv], V[a + 2*Nv]);
        d3 rb = make_d3(V[b + 0*Nv], V[b + 1*Nv], V[b + 2*Nv]);
        c[e] = make_d3(0.5*(ra.x+rb.x), 0.5*(ra.y+rb.y), 0.5*(ra.z+rb.z));
        if(c[e].z<zmin) zmin=c[e].z;
        if(c[e].z>zmax) zmax=c[e].z;
    }
    double zmid = 0.5*(zmin+zmax);

    int e0 = edge0_1based - 1;
    int layer0 = (c[e0].z > zmid) ? +1 : -1;
    if(layer_out) *layer_out = layer0;

    d3 r0 = c[e0];

    double sumw = 0.0;
    for(mwSize e=0;e<Ne;++e){
        int layer = (c[e].z > zmid) ? +1 : -1;
        if(layer != layer0){
            w_out[e]=0.0;
            continue;
        }
        d3 d = sub3(c[e], r0);
        double dist = norm3(d);
        double w = exp(-(dist/spread)*(dist/spread));
        w_out[e] = w;
        sumw += w;
    }
    if(sumw < 1e-30) mexErrMsgIdAndTxt("mex:rhs","spreadRadius too small (no weights).");

    for(mwSize e=0;e<Ne;++e) w_out[e] /= sumw;

    mxFree(c);
}

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    // Signature:
    // [Zin,S11] = patch_sweep_fastquad_cuda( ...
    //   V, F(int32), E(int32),
    //   plusTri(int32), minusTri(int32), plusSign(int32), minusSign(int32),
    //   rp, rm, Ap, Am, len,
    //   prepack(struct),
    //   freqs,
    //   mu, eps,
    //   patchEdge, groundEdge, V0, spreadRadius, Z0 )

    check(nrhs==21, "mex:args", "Expected 21 inputs.");
    check(nlhs==2,  "mex:args", "Expected 2 outputs [Zin,S11].");

    const mxArray* Vmx = prhs[0];
    const mxArray* Fmx = prhs[1];
    const mxArray* Emx = prhs[2];

    const mxArray* plusTriMx   = prhs[3];
    const mxArray* minusTriMx  = prhs[4];
    const mxArray* plusSignMx  = prhs[5];
    const mxArray* minusSignMx = prhs[6];

    const mxArray* rpMx  = prhs[7];
    const mxArray* rmMx  = prhs[8];
    const mxArray* ApMx  = prhs[9];
    const mxArray* AmMx  = prhs[10];
    const mxArray* lenMx = prhs[11];

    const mxArray* prepackMx = prhs[12];
    const mxArray* freqsMx   = prhs[13];

    double mu = mxGetScalar(prhs[14]);
    cuDoubleComplex eps_c = get_complex_scalar(prhs[15]);

    int patchEdge  = (int)mxGetScalar(prhs[16]);
    int groundEdge = (int)mxGetScalar(prhs[17]);
    double V0      = mxGetScalar(prhs[18]);
    double spread  = mxGetScalar(prhs[19]);
    double Z0      = mxGetScalar(prhs[20]);


    // checks
    check(mxIsDouble(Vmx) && mxGetN(Vmx)==3, "mex:V", "V must be [Nv x 3] double.");
    check(mxIsInt32(Fmx) && mxGetN(Fmx)==3, "mex:F", "F must be [Nt x 3] int32.");
    check(mxIsInt32(Emx) && mxGetN(Emx)==2, "mex:E", "E must be [Ne x 2] int32.");

    mwSize Nv = mxGetM(Vmx);
    mwSize Nt = mxGetM(Fmx);
    mwSize Ne = mxGetM(Emx);

    check(mxIsDouble(freqsMx), "mex:freqs", "freqs must be double vector.");
    mwSize Nf = mxGetNumberOfElements(freqsMx);

    check(mxIsStruct(prepackMx), "mex:prepack", "prepack must be struct.");

    check(patchEdge>=1 && patchEdge<=(int)Ne, "mex:port", "patchEdge out of range.");
    check(groundEdge>=1 && groundEdge<=(int)Ne, "mex:port", "groundEdge out of range.");

    const double* Vh = mxGetPr(Vmx);
    const int* Eh    = (const int*)mxGetData(Emx);

    const int* plusTri   = (const int*)mxGetData(plusTriMx);
    const int* minusTri  = (const int*)mxGetData(minusTriMx);
    const int* plusSign  = (const int*)mxGetData(plusSignMx);
    const int* minusSign = (const int*)mxGetData(minusSignMx);

    const double* rpH = mxGetPr(rpMx);
    const double* rmH = mxGetPr(rmMx);
    const double* ApH = mxGetPr(ApMx);
    const double* AmH = mxGetPr(AmMx);
    const double* lenH= mxGetPr(lenMx);

    const double* freqs = mxGetPr(freqsMx);

    auto getField = [&](const char* name)->const mxArray*{
        const mxArray* f = mxGetField(prepackMx, 0, name);
        check(f!=nullptr, "mex:prepack", "Missing field in prepack.");
        return f;
    };

    const mxArray* r_reg_mx   = getField("r_reg");
    const mxArray* w_reg_mx   = getField("w_reg");
    const mxArray* nearMat_mx = getField("nearMat");

    const mxArray* self_r_mx  = getField("self_r");
    const mxArray* self_w_mx  = getField("self_w");
    const mxArray* self_a_mx  = getField("self_a");

    const mxArray* near_r_mx  = getField("near_r");
    const mxArray* near_w_mx  = getField("near_w");

    const mxArray* aeqTri_mx  = getField("aeqTri");
    const mxArray* near_alpha_mx = getField("near_alpha");

    double near_alpha = mxGetScalar(near_alpha_mx);

    const mwSize* rreg_dims = mxGetDimensions(r_reg_mx);
    int Nq  = (int)rreg_dims[0];

    const mwSize* selfr_dims = mxGetDimensions(self_r_mx);
    int Nq2 = (int)selfr_dims[0];
    int NsS = (int)selfr_dims[2];

    const mwSize* nearr_dims = mxGetDimensions(near_r_mx);
    int NsN = (int)nearr_dims[2];

    const double* r_reg  = mxGetPr(r_reg_mx);
    const double* w_reg  = mxGetPr(w_reg_mx);
    const uint8_t* nearMat = (const uint8_t*)mxGetData(nearMat_mx);

    const double* self_r = mxGetPr(self_r_mx);
    const double* self_w = mxGetPr(self_w_mx);
    const double* self_a = mxGetPr(self_a_mx);

    const double* near_r = mxGetPr(near_r_mx);
    const double* near_w = mxGetPr(near_w_mx);

    const double* aeqTri = mxGetPr(aeqTri_mx);

    // ---- Build RHS weights on CPU once (CPU-Fast ile aynı mantık)
    double* wPatch = (double*)mxMalloc(Ne*sizeof(double));
    double* wGnd   = (double*)mxMalloc(Ne*sizeof(double));
    build_weights_gaussian(Vh, Nv, Eh, Ne, patchEdge, spread, wPatch, nullptr);
    build_weights_gaussian(Vh, Nv, Eh, Ne, groundEdge, spread, wGnd, nullptr);

    // RHS vector (real) once: +V0/2*wPatch - V0/2*wGnd
    cuDoubleComplex* b_host = (cuDoubleComplex*)mxMalloc(Ne*sizeof(cuDoubleComplex));
    for(mwSize i=0;i<Ne;i++){
        double vr = (+V0/2.0)*wPatch[i] + (-V0/2.0)*wGnd[i];
        b_host[i] = make_cuDoubleComplex(vr, 0.0);
    }

    // rp/rm AoS
    d3* rpA = (d3*)mxMalloc(Ne*sizeof(d3));
    d3* rmA = (d3*)mxMalloc(Ne*sizeof(d3));
    for(mwSize e=0;e<Ne;e++){
        rpA[e] = make_d3(rpH[e + 0*Ne], rpH[e + 1*Ne], rpH[e + 2*Ne]);
        rmA[e] = make_d3(rmH[e + 0*Ne], rmH[e + 1*Ne], rmH[e + 2*Ne]);
    }

    // ---- Device alloc/copy (ONE TIME)
    int *d_plusTri,*d_minusTri,*d_plusSign,*d_minusSign;
    d3 *d_rp,*d_rm;
    double *d_Ap,*d_Am,*d_len;

    double *d_rreg,*d_wreg;
    uint8_t *d_nearMat;

    double *d_selfr,*d_selfw,*d_selfa;
    double *d_nearr,*d_nearw;
    double *d_aeq;

    cuDoubleComplex *d_A, *d_b, *d_x;

    CUDA_OK(cudaMalloc(&d_plusTri,   Ne*sizeof(int)));
    CUDA_OK(cudaMalloc(&d_minusTri,  Ne*sizeof(int)));
    CUDA_OK(cudaMalloc(&d_plusSign,  Ne*sizeof(int)));
    CUDA_OK(cudaMalloc(&d_minusSign, Ne*sizeof(int)));

    CUDA_OK(cudaMalloc(&d_rp, Ne*sizeof(d3)));
    CUDA_OK(cudaMalloc(&d_rm, Ne*sizeof(d3)));

    CUDA_OK(cudaMalloc(&d_Ap,  Ne*sizeof(double)));
    CUDA_OK(cudaMalloc(&d_Am,  Ne*sizeof(double)));
    CUDA_OK(cudaMalloc(&d_len, Ne*sizeof(double)));

    CUDA_OK(cudaMemcpy(d_plusTri, plusTri, Ne*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_minusTri, minusTri, Ne*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_plusSign, plusSign, Ne*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_minusSign, minusSign, Ne*sizeof(int), cudaMemcpyHostToDevice));

    CUDA_OK(cudaMemcpy(d_rp, rpA, Ne*sizeof(d3), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_rm, rmA, Ne*sizeof(d3), cudaMemcpyHostToDevice));

    CUDA_OK(cudaMemcpy(d_Ap, ApH, Ne*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_Am, AmH, Ne*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_len, lenH, Ne*sizeof(double), cudaMemcpyHostToDevice));

    // pre arrays
    size_t sz_rreg = (size_t)Nq*3*(size_t)Nt*sizeof(double);
    size_t sz_wreg = (size_t)Nq*(size_t)Nt*sizeof(double);
    size_t sz_nearMat = (size_t)Nt*(size_t)Nt*sizeof(uint8_t);

    size_t sz_selfr = (size_t)Nq2*3*(size_t)NsS*(size_t)Nt*sizeof(double);
    size_t sz_selfw = (size_t)Nq2*(size_t)NsS*(size_t)Nt*sizeof(double);
    size_t sz_selfa = (size_t)NsS*(size_t)Nt*sizeof(double);

    size_t sz_nearr = (size_t)Nq2*3*(size_t)NsN*(size_t)Nt*sizeof(double);
    size_t sz_nearw = (size_t)Nq2*(size_t)NsN*(size_t)Nt*sizeof(double);

    size_t sz_aeq = (size_t)Nt*sizeof(double);

    CUDA_OK(cudaMalloc(&d_rreg, sz_rreg));
    CUDA_OK(cudaMalloc(&d_wreg, sz_wreg));
    CUDA_OK(cudaMalloc(&d_nearMat, sz_nearMat));
    CUDA_OK(cudaMemcpy(d_rreg, r_reg, sz_rreg, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_wreg, w_reg, sz_wreg, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_nearMat, nearMat, sz_nearMat, cudaMemcpyHostToDevice));

    CUDA_OK(cudaMalloc(&d_selfr, sz_selfr));
    CUDA_OK(cudaMalloc(&d_selfw, sz_selfw));
    CUDA_OK(cudaMalloc(&d_selfa, sz_selfa));
    CUDA_OK(cudaMemcpy(d_selfr, self_r, sz_selfr, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_selfw, self_w, sz_selfw, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_selfa, self_a, sz_selfa, cudaMemcpyHostToDevice));

    CUDA_OK(cudaMalloc(&d_nearr, sz_nearr));
    CUDA_OK(cudaMalloc(&d_nearw, sz_nearw));
    CUDA_OK(cudaMemcpy(d_nearr, near_r, sz_nearr, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_nearw, near_w, sz_nearw, cudaMemcpyHostToDevice));

    CUDA_OK(cudaMalloc(&d_aeq, sz_aeq));
    CUDA_OK(cudaMemcpy(d_aeq, aeqTri, sz_aeq, cudaMemcpyHostToDevice));

    // Z and RHS buffers
    CUDA_OK(cudaMalloc(&d_A, (size_t)Ne*(size_t)Ne*sizeof(cuDoubleComplex)));
    CUDA_OK(cudaMalloc(&d_b, Ne*sizeof(cuDoubleComplex)));
    CUDA_OK(cudaMalloc(&d_x, Ne*sizeof(cuDoubleComplex)));
    CUDA_OK(cudaMemcpy(d_b, b_host, Ne*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

    // ---- cuSOLVER init
    cusolverDnHandle_t solver;
    CUSOLVER_OK(cusolverDnCreate(&solver));

    int* d_ipiv; int* d_info;
    CUDA_OK(cudaMalloc(&d_ipiv, Ne*sizeof(int)));
    CUDA_OK(cudaMalloc(&d_info, sizeof(int)));

    int lwork=0;
    cuDoubleComplex* d_work=nullptr;

    // outputs
    plhs[0] = mxCreateDoubleMatrix((mwSize)Nf, 1, mxCOMPLEX);
    plhs[1] = mxCreateDoubleMatrix((mwSize)Nf, 1, mxCOMPLEX);

#if MX_HAS_INTERLEAVED_COMPLEX
    mxComplexDouble* ZinOut = mxGetComplexDoubles(plhs[0]);
    mxComplexDouble* S11Out = mxGetComplexDoubles(plhs[1]);
#else
    double* ZinRe = mxGetPr(plhs[0]); double* ZinIm = mxGetPi(plhs[0]);
    double* S11Re = mxGetPr(plhs[1]); double* S11Im = mxGetPi(plhs[1]);
#endif

    // host solution buffer (CPU'ya geri getirip port akımı hesaplayacağız)
    cuDoubleComplex* x_host = (cuDoubleComplex*)mxMalloc(Ne*sizeof(cuDoubleComplex));

    dim3 block(8,8);
    dim3 grid((unsigned)((Ne + block.x - 1)/block.x),
              (unsigned)((Ne + block.y - 1)/block.y));

    for(mwSize fi=0; fi<Nf; ++fi){
        double fHz = freqs[fi];
        double omega = 2.0*PI_D*fHz;

        // k = omega * sqrt(mu*eps)
        // eps complex: (er + j ei)
        // sqrt complex on host (manual)
        // use std::complex? (avoid include) -> simple polar:
        double er = cuCreal(eps_c);
        double ei = cuCimag(eps_c);

        // mu*eps
        double ar = mu*er;
        double ai = mu*ei;

        double r = sqrt(ar*ar + ai*ai);
        double ang = atan2(ai, ar);
        double sr = sqrt(r);
        double half = 0.5*ang;

        double sqr = sr*cos(half);
        double sqi = sr*sin(half);

        cuDoubleComplex k_c = make_cuDoubleComplex(omega*sqr, omega*sqi);

        cuDoubleComplex alpha = make_cuDoubleComplex(0.0, omega*mu);

        // beta = 1/(j*omega*eps)
        cuDoubleComplex denom = make_cuDoubleComplex(-omega*ei, omega*er);
        double denom_abs2 = cuCreal(denom)*cuCreal(denom) + cuCimag(denom)*cuCimag(denom);
        cuDoubleComplex beta = make_cuDoubleComplex(cuCreal(denom)/denom_abs2, -cuCimag(denom)/denom_abs2);

        // Assemble Z on GPU
        kernel_Z_fastquad<<<grid,block>>>(
            d_A, (int)Ne,
            d_plusTri, d_minusTri, d_plusSign, d_minusSign,
            d_rp, d_rm, d_Ap, d_Am, d_len,
            d_rreg, d_wreg, Nq, (int)Nt,
            d_nearMat,
            d_selfr, d_selfw, d_selfa, Nq2, NsS,
            d_nearr, d_nearw, NsN,
            d_aeq, near_alpha,
            k_c, alpha, beta
        );
        CUDA_OK(cudaGetLastError());
        CUDA_OK(cudaDeviceSynchronize());

        // x = b (copy RHS each frequency because getrs overwrites RHS)
        CUDA_OK(cudaMemcpy(d_x, d_b, Ne*sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice));

        // workspace query once
        if(d_work==nullptr){
            CUSOLVER_OK(cusolverDnZgetrf_bufferSize(solver, (int)Ne, (int)Ne, d_A, (int)Ne, &lwork));
            CUDA_OK(cudaMalloc(&d_work, (size_t)lwork*sizeof(cuDoubleComplex)));
        }

        // LU factorization + solve
        CUSOLVER_OK(cusolverDnZgetrf(solver, (int)Ne, (int)Ne, d_A, (int)Ne, d_work, d_ipiv, d_info));
        CUSOLVER_OK(cusolverDnZgetrs(solver, CUBLAS_OP_N, (int)Ne, 1, d_A, (int)Ne, d_ipiv, d_x, (int)Ne, d_info));
        CUDA_OK(cudaDeviceSynchronize());

        // ---- CPU’ya geri getir (senin istediğin gibi)
        CUDA_OK(cudaMemcpy(x_host, d_x, Ne*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));

        // port current (CPU FAST ile aynı: sum( (wPatch - wGnd).*I.*len ))
        double Ipr=0.0, Ipi=0.0;
        for(mwSize i=0;i<Ne;i++){
            double wr = (wPatch[i] - wGnd[i]) * lenH[i];
            double xr = cuCreal(x_host[i]);
            double xi = cuCimag(x_host[i]);
            Ipr += wr * xr;
            Ipi += wr * xi;
        }

        // Zin = V0 / Iport
        // (V0 real)
        double den = Ipr*Ipr + Ipi*Ipi;
        double Zin_r =  V0 * ( Ipr / den);
        double Zin_i = -V0 * ( Ipi / den);

        // S11 = (Zin - Z0)/(Zin + Z0)
        // complex division
        double a_r = (Zin_r - Z0), a_i = Zin_i;
        double b_r = (Zin_r + Z0), b_i = Zin_i;
        double bden = b_r*b_r + b_i*b_i;
        double S11_r = (a_r*b_r + a_i*b_i)/bden;
        double S11_i = (a_i*b_r - a_r*b_i)/bden;

#if MX_HAS_INTERLEAVED_COMPLEX
        ZinOut[fi].real = Zin_r; ZinOut[fi].imag = Zin_i;
        S11Out[fi].real = S11_r; S11Out[fi].imag = S11_i;
#else
        ZinRe[fi]=Zin_r; ZinIm[fi]=Zin_i;
        S11Re[fi]=S11_r; S11Im[fi]=S11_i;
#endif
    }

    // cleanup
    mxFree(x_host);
    mxFree(rpA); mxFree(rmA);
    mxFree(wPatch); mxFree(wGnd);
    mxFree(b_host);

    if(d_work) CUDA_OK(cudaFree(d_work));
    CUSOLVER_OK(cusolverDnDestroy(solver));

    CUDA_OK(cudaFree(d_A));
    CUDA_OK(cudaFree(d_b));
    CUDA_OK(cudaFree(d_x));
    CUDA_OK(cudaFree(d_ipiv));
    CUDA_OK(cudaFree(d_info));

    CUDA_OK(cudaFree(d_plusTri));
    CUDA_OK(cudaFree(d_minusTri));
    CUDA_OK(cudaFree(d_plusSign));
    CUDA_OK(cudaFree(d_minusSign));
    CUDA_OK(cudaFree(d_rp));
    CUDA_OK(cudaFree(d_rm));
    CUDA_OK(cudaFree(d_Ap));
    CUDA_OK(cudaFree(d_Am));
    CUDA_OK(cudaFree(d_len));

    CUDA_OK(cudaFree(d_rreg));
    CUDA_OK(cudaFree(d_wreg));
    CUDA_OK(cudaFree(d_nearMat));
    CUDA_OK(cudaFree(d_selfr));
    CUDA_OK(cudaFree(d_selfw));
    CUDA_OK(cudaFree(d_selfa));
    CUDA_OK(cudaFree(d_nearr));
    CUDA_OK(cudaFree(d_nearw));
    CUDA_OK(cudaFree(d_aeq));
}
