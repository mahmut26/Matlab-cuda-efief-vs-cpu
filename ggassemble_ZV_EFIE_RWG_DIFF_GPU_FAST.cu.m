// assemble_ZV_EFIE_RWG_DIFF_GPU_FAST_QUAD.cu
#include "mex.h"
#include "matrix.h"

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <math.h>
#include <stdint.h>

#ifndef PI_D
#define PI_D 3.141592653589793238462643383279502884
#endif

#define CUDA_OK(call) do{ cudaError_t e=(call); if(e!=cudaSuccess){ mexErrMsgIdAndTxt("cuda:rt", cudaGetErrorString(e)); } }while(0)

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

// ----- RWG eval (GPU) -----
// fm(r) and divm on triangle t, for edge m
// using same formula as MATLAB rwg_eval_on_triangle
__device__ static inline void rwg_eval_point(
    int t, int tp, int tm,
    int plusSign, int minusSign,
    double len, double Ap, double Am,
    d3 rp, d3 rm,
    d3 r,
    d3 &f_out,
    double &div_out
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
    // shouldn't happen in Galerkin for those tm/tn choices, but safe:
    f_out = make_d3(0,0,0);
    div_out = 0.0;
}

// ===========================
// REGULAR integral (cached quad) - GPU
// r_reg: [Nq x 3 x Nt]  (MATLAB column-major)
// w_reg: [Nq x Nt]
// ===========================
__device__ static inline void integral_regular(
    const double* r_reg, const double* w_reg,
    int Nq, int Nt,
    // rwg
    int m, int n,
    int t1, int t2,
    const int* plusTri, const int* minusTri,
    const int* plusSign, const int* minusSign,
    const double* rp, const double* rm,
    const double* Ap, const double* Am,
    const double* len,
    cuDoubleComplex k,
    cuDoubleComplex &Ivec, cuDoubleComplex &Isca
){
    // NOTE: t1,t2 are 1-based triangle indices
    int t1z = t1-1;
    int t2z = t2-1;

    int tp_m = plusTri[m], tm_m = minusTri[m];
    int tp_n = plusTri[n], tm_n = minusTri[n];

    d3 rp_m = make_d3(rp[m + 0], rp[m + 1], rp[m + 2]); // WRONG indexing if treated raw
    // We'll pass rp/rm as AoS on device, so we won't use this path.
    // To keep code short, regular/near/self will be called with AoS rp/rm arrays below.
    (void)rp_m; (void)tm_m; (void)tp_m; (void)tp_n; (void)tm_n; (void)t1z; (void)t2z;
    (void)minusSign; (void)plusSign; (void)Ap; (void)Am; (void)len; (void)r_reg; (void)w_reg; (void)Nq; (void)Nt; (void)k;
    Ivec = make_cuDoubleComplex(0,0);
    Isca = make_cuDoubleComplex(0,0);
}

// Biz rp/rm için AoS (d3) kullanacağız; yukarıdaki stub’ı kullanmıyoruz.
// Aşağıda gerçek, AoS kullanan versiyon var:

__device__ static inline void integral_regular_AoS(
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

    d3 rp_m = rp[m];
    d3 rm_m = rm[m];
    d3 rp_n = rp[n];
    d3 rm_n = rm[n];

    double lm = len[m];
    double ln = len[n];

    double Apm = Ap[m], Amm = Am[m];
    double Apn = Ap[n], Amn = Am[n];

    int sppm = plusSign[m], smpm = minusSign[m];
    int sppn = plusSign[n], smpn = minusSign[n];

    cuDoubleComplex Ivec_sum = make_cuDoubleComplex(0,0);
    cuDoubleComplex Isca_sum = make_cuDoubleComplex(0,0);

    // loops over quadrature points
    // r_reg layout: r_reg(i,xyz,t) in MATLAB => index = i + Nq*xyz + Nq*3*t
    // w_reg layout: w_reg(i,t) => i + Nq*t
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
            double dotff = dot3(fm, fn);
            double divdiv = divm * divn;

            // Ivec += w*dotff*G
            // Isca += w*divdiv*G
            Ivec_sum = cadd(Ivec_sum, cmul_real(G, w*dotff));
            Isca_sum = cadd(Isca_sum, cmul_real(G, w*divdiv));
        }
    }

    Ivec = Ivec_sum;
    Isca = Isca_sum;
}

// ===========================
// NEAR integral (cached subdiv quad) - GPU
// near_r: [Nq2 x 3 x NsN x Nt]
// near_w: [Nq2 x NsN x Nt]
// softening: a_soft = near_alpha * min(aeqTri(t1), aeqTri(t2))
// ===========================
__device__ static inline void integral_near_cached(
    const double* near_r, const double* near_w,
    const double* aeqTri,
    double near_alpha,
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

    double aeq = aeqTri[t1z] < aeqTri[t2z] ? aeqTri[t1z] : aeqTri[t2z];
    double a_soft = near_alpha * aeq;

    int tp_m = plusTri[m], tm_m = minusTri[m];
    int tp_n = plusTri[n], tm_n = minusTri[n];

    d3 rp_m = rp[m];
    d3 rm_m = rm[m];
    d3 rp_n = rp[n];
    d3 rm_n = rm[n];

    double lm = len[m];
    double ln = len[n];

    double Apm = Ap[m], Amm = Am[m];
    double Apn = Ap[n], Amn = Am[n];

    int sppm = plusSign[m], smpm = minusSign[m];
    int sppn = plusSign[n], smpn = minusSign[n];

    cuDoubleComplex Ivec_sum = make_cuDoubleComplex(0,0);
    cuDoubleComplex Isca_sum = make_cuDoubleComplex(0,0);

    // indexing:
    // near_r(i,xyz,p,t) => i + Nq2*xyz + Nq2*3*p + Nq2*3*NsN*t
    // near_w(i,p,t) => i + Nq2*p + Nq2*NsN*t
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
                    // near: sqrt(R^2 + a_soft^2)
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

// ===========================
// SELF integral (cached subdiv quad) - GPU
// self_r: [Nq2 x 3 x NsS x Nt]
// self_w: [Nq2 x NsS x Nt]
// self_a: [NsS x Nt]  (per-subtriangle aSoft)
// rule: if p==q: sqrt(R^2 + aS(p)^2) else max(R,1e-15)
// ===========================
__device__ static inline void integral_self_cached(
    const double* self_r, const double* self_w, const double* self_a,
    int Nq2, int NsS, int Nt,
    int m, int n,
    int t, // t1==t2==t
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

    d3 rp_m = rp[m];
    d3 rm_m = rm[m];
    d3 rp_n = rp[n];
    d3 rm_n = rm[n];

    double lm = len[m];
    double ln = len[n];

    double Apm = Ap[m], Amm = Am[m];
    double Apn = Ap[n], Amn = Am[n];

    int sppm = plusSign[m], smpm = minusSign[m];
    int sppn = plusSign[n], smpn = minusSign[n];

    cuDoubleComplex Ivec_sum = make_cuDoubleComplex(0,0);
    cuDoubleComplex Isca_sum = make_cuDoubleComplex(0,0);

    // indexing:
    // self_r(i,xyz,p,t) => i + Nq2*xyz + Nq2*3*p + Nq2*3*NsS*t
    // self_w(i,p,t) => i + Nq2*p + Nq2*NsS*t
    // self_a(p,t) => p + NsS*t
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

// ===========================
// Kernel: build Z (Galerkin) with cached rules
// ===========================
__global__ void kernel_Z_fastquad(
    cuDoubleComplex* Z, int Ne,
    // rwg
    const int* plusTri, const int* minusTri,
    const int* plusSign, const int* minusSign,
    const d3* rp, const d3* rm,
    const double* Ap, const double* Am,
    const double* len,
    // pre
    const double* r_reg, const double* w_reg, int Nq, int Nt,
    const uint8_t* nearMat, // Nt x Nt
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

    int tm1 = plusTri[m];
    int tm2 = minusTri[m];
    int tn1 = plusTri[n];
    int tn2 = minusTri[n];

    int tm[2] = { tm1, tm2 };
    int tn[2] = { tn1, tn2 };

    cuDoubleComplex Zmn = make_cuDoubleComplex(0,0);

    for(int im=0; im<2; ++im){
        int t1 = tm[im];
        if(t1==0) continue;

        for(int in=0; in<2; ++in){
            int t2 = tn[in];
            if(t2==0) continue;

            cuDoubleComplex Ivec, Isca;

            if(t1 == t2){
                integral_self_cached(self_r, self_w, self_a, Nq2, NsS, Nt,
                                     m, n, t1,
                                     plusTri, minusTri, plusSign, minusSign,
                                     rp, rm, Ap, Am, len, k, Ivec, Isca);
            }else{
                // nearMat is [Nt x Nt], MATLAB column-major => idx = (t1-1) + Nt*(t2-1)
                int idx = (t1-1) + Nt*(t2-1);
                if(nearMat[idx]){
                    integral_near_cached(near_r, near_w, aeqTri, near_alpha,
                                         Nq2, NsN, Nt,
                                         m, n, t1, t2,
                                         plusTri, minusTri, plusSign, minusSign,
                                         rp, rm, Ap, Am, len, k, Ivec, Isca);
                }else{
                    integral_regular_AoS(r_reg, w_reg, Nq, Nt,
                                         m, n, t1, t2,
                                         plusTri, minusTri, plusSign, minusSign,
                                         rp, rm, Ap, Am, len, k, Ivec, Isca);
                }
            }

            // Zmn += alpha*Ivec + beta*Isca
            Zmn = cadd(Zmn, cadd(cmul(alpha, Ivec), cmul(beta, Isca)));
        }
    }

    Z[m + n*Ne] = Zmn; // column-major
}

// ============ host: Vrhs gaussian (CPU) ============
static void build_Vrhs_gaussian(
    const double* V, mwSize Nv,
    const int* E, mwSize Ne,
    int patchEdge, int groundEdge,
    double V0, double spread,
    double* Vrhs_out // real
){
    // edge centers
    d3* ecent = (d3*)mxMalloc(Ne*sizeof(d3));
    double zmin=1e100, zmax=-1e100;
    for(mwSize e=0; e<Ne; ++e){
        int a = E[e + 0*Ne] - 1;
        int b = E[e + 1*Ne] - 1;
        d3 ra = make_d3(V[a + 0*Nv], V[a + 1*Nv], V[a + 2*Nv]);
        d3 rb = make_d3(V[b + 0*Nv], V[b + 1*Nv], V[b + 2*Nv]);
        d3 c  = make_d3(0.5*(ra.x+rb.x), 0.5*(ra.y+rb.y), 0.5*(ra.z+rb.z));
        ecent[e]=c;
        if(c.z<zmin) zmin=c.z;
        if(c.z>zmax) zmax=c.z;
    }
    double zmid = 0.5*(zmin+zmax);

    double* tmp = (double*)mxMalloc(Ne*sizeof(double));
    for(mwSize e=0;e<Ne;++e) Vrhs_out[e]=0.0;

    auto add_gauss = [&](int edge1based, double amp){
        int e0 = edge1based-1;
        int layer0 = (ecent[e0].z > zmid) ? +1 : -1;
        d3 r0 = ecent[e0];
        double sumw = 0.0;
        for(mwSize e=0;e<Ne;++e){
            int layer = (ecent[e].z > zmid) ? +1 : -1;
            if(layer!=layer0){ tmp[e]=0.0; continue; }
            d3 d = sub3(ecent[e], r0);
            double dist = norm3(d);
            double w = exp(-(dist/spread)*(dist/spread));
            tmp[e]=w;
            sumw += w;
        }
        if(sumw < 1e-30) mexErrMsgIdAndTxt("mex:rhs","spreadRadius too small (no weights).");
        for(mwSize e=0;e<Ne;++e) Vrhs_out[e] += amp*(tmp[e]/sumw);
    };

    add_gauss(patchEdge,  +V0/2.0);
    add_gauss(groundEdge, -V0/2.0);

    mxFree(tmp);
    mxFree(ecent);
}

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    check(nrhs==18, "mex:args",
          "Expected 18 inputs: V,F,E,plusTri,minusTri,plusSign,minusSign,rp,rm,Ap,Am,len,prepack,k,omega,mu,eps,excvec");
    check(nlhs==2, "mex:args", "Expected 2 outputs [Z,Vrhs].");

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

    cuDoubleComplex k_c   = get_complex_scalar(prhs[13]);
    double omega          = mxGetScalar(prhs[14]);
    double mu             = mxGetScalar(prhs[15]);
    cuDoubleComplex eps_c = get_complex_scalar(prhs[16]);

    const mxArray* excMx = prhs[17];

    check(mxIsDouble(Vmx) && mxGetN(Vmx)==3, "mex:V", "V must be [Nv x 3] double.");
    check(mxIsInt32(Fmx)  && mxGetN(Fmx)==3, "mex:F", "F must be [Nt x 3] int32.");
    check(mxIsInt32(Emx)  && mxGetN(Emx)==2, "mex:E", "E must be [Ne x 2] int32.");

    mwSize Nv = mxGetM(Vmx);
    mwSize Nt = mxGetM(Fmx);
    mwSize Ne = mxGetM(Emx);

    check(mxIsStruct(prepackMx), "mex:prepack", "prepack must be a struct from efie_precompute_pack.");

    // excvec = [patchEdge, groundEdge, V0, spreadRadius]
    check(mxIsDouble(excMx) && mxGetNumberOfElements(excMx)>=4, "mex:exc", "excvec must be [patchEdge, groundEdge, V0, spreadRadius].");
    const double* excv = mxGetPr(excMx);
    int patchEdge  = (int)excv[0];
    int groundEdge = (int)excv[1];
    double V0      = excv[2];
    double spread  = excv[3];

    check(patchEdge>=1 && patchEdge<=(int)Ne, "mex:port", "patchEdge out of range.");
    check(groundEdge>=1 && groundEdge<=(int)Ne, "mex:port", "groundEdge out of range.");

    // host arrays
    const double* Vh = mxGetPr(Vmx);
    const int*    Eh = (const int*)mxGetData(Emx);

    const int* plusTri   = (const int*)mxGetData(plusTriMx);
    const int* minusTri  = (const int*)mxGetData(minusTriMx);
    const int* plusSign  = (const int*)mxGetData(plusSignMx);
    const int* minusSign = (const int*)mxGetData(minusSignMx);

    // rp/rm input is [Ne x 3] double in MATLAB column-major
    const double* rpH = mxGetPr(rpMx);
    const double* rmH = mxGetPr(rmMx);

    const double* ApH  = mxGetPr(ApMx);
    const double* AmH  = mxGetPr(AmMx);
    const double* lenH = mxGetPr(lenMx);

    // ---- pull packed pre fields ----
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

    check(mxIsDouble(r_reg_mx) && mxGetNumberOfDimensions(r_reg_mx)==3, "mex:r_reg", "r_reg must be [Nq x 3 x Nt] double.");
    check(mxIsDouble(w_reg_mx) && mxGetNumberOfDimensions(w_reg_mx)==2, "mex:w_reg", "w_reg must be [Nq x Nt] double.");
    check(mxIsUint8(nearMat_mx), "mex:nearMat", "nearMat must be uint8 [Nt x Nt].");

    check(mxIsDouble(self_r_mx) && mxGetNumberOfDimensions(self_r_mx)==4, "mex:self_r", "self_r must be [Nq2 x 3 x NsS x Nt].");
    check(mxIsDouble(self_w_mx) && mxGetNumberOfDimensions(self_w_mx)==3, "mex:self_w", "self_w must be [Nq2 x NsS x Nt].");
    check(mxIsDouble(self_a_mx) && mxGetNumberOfDimensions(self_a_mx)==2, "mex:self_a", "self_a must be [NsS x Nt].");

    check(mxIsDouble(near_r_mx) && mxGetNumberOfDimensions(near_r_mx)==4, "mex:near_r", "near_r must be [Nq2 x 3 x NsN x Nt].");
    check(mxIsDouble(near_w_mx) && mxGetNumberOfDimensions(near_w_mx)==3, "mex:near_w", "near_w must be [Nq2 x NsN x Nt].");

    check(mxIsDouble(aeqTri_mx) && mxGetNumberOfElements(aeqTri_mx)==Nt, "mex:aeqTri", "aeqTri must be [Nt x 1].");
    double near_alpha = mxGetScalar(near_alpha_mx);

    const mwSize* rreg_dims = mxGetDimensions(r_reg_mx);
    int Nq = (int)rreg_dims[0];
    check((int)rreg_dims[1]==3, "mex:r_reg", "r_reg dim2 must be 3.");
    check((mwSize)rreg_dims[2]==Nt, "mex:r_reg", "r_reg dim3 must be Nt.");

    const mwSize* selfr_dims = mxGetDimensions(self_r_mx);
    int Nq2 = (int)selfr_dims[0];
    check((int)selfr_dims[1]==3, "mex:self_r", "self_r dim2 must be 3.");
    int NsS = (int)selfr_dims[2];
    check((mwSize)selfr_dims[3]==Nt, "mex:self_r", "self_r dim4 must be Nt.");

    const mwSize* nearr_dims = mxGetDimensions(near_r_mx);
    check((int)nearr_dims[0]==Nq2, "mex:near_r", "near_r Nq2 mismatch.");
    int NsN = (int)nearr_dims[2];
    check((mwSize)nearr_dims[3]==Nt, "mex:near_r", "near_r dim4 must be Nt.");

    const double* r_reg   = mxGetPr(r_reg_mx);
    const double* w_reg   = mxGetPr(w_reg_mx);
    const uint8_t* nearMat = (const uint8_t*)mxGetData(nearMat_mx);

    const double* self_r  = mxGetPr(self_r_mx);
    const double* self_w  = mxGetPr(self_w_mx);
    const double* self_a  = mxGetPr(self_a_mx);

    const double* near_r  = mxGetPr(near_r_mx);
    const double* near_w  = mxGetPr(near_w_mx);

    const double* aeqTri  = mxGetPr(aeqTri_mx);

    // ---- Build Vrhs on host (same as CPU gaussian)
    double* Vrhs_r = (double*)mxMalloc(Ne*sizeof(double));
    build_Vrhs_gaussian(Vh, Nv, Eh, Ne, patchEdge, groundEdge, V0, spread, Vrhs_r);

    // ---- alpha/beta
    // alpha = j*omega*mu
    cuDoubleComplex alpha = make_cuDoubleComplex(0.0, omega*mu);

    // beta  = 1/(j*omega*eps)
    double er = cuCreal(eps_c);
    double ei = cuCimag(eps_c);
    cuDoubleComplex denom = make_cuDoubleComplex(-omega*ei, omega*er);
    double denom_abs2 = cuCreal(denom)*cuCreal(denom) + cuCimag(denom)*cuCimag(denom);
    check(denom_abs2>0.0, "mex:eps", "eps causes denom=0.");
    cuDoubleComplex beta = make_cuDoubleComplex(cuCreal(denom)/denom_abs2, -cuCimag(denom)/denom_abs2);

    // ---- pack rp/rm into AoS for device
    d3* rpA = (d3*)mxMalloc(Ne*sizeof(d3));
    d3* rmA = (d3*)mxMalloc(Ne*sizeof(d3));
    for(mwSize e=0;e<Ne;++e){
        rpA[e] = make_d3(rpH[e + 0*Ne], rpH[e + 1*Ne], rpH[e + 2*Ne]);
        rmA[e] = make_d3(rmH[e + 0*Ne], rmH[e + 1*Ne], rmH[e + 2*Ne]);
    }

    // ---- device alloc/copy
    int *d_plusTri,*d_minusTri,*d_plusSign,*d_minusSign;
    d3 *d_rp,*d_rm;
    double *d_Ap,*d_Am,*d_len;

    double *d_rreg,*d_wreg;
    uint8_t *d_nearMat;

    double *d_selfr,*d_selfw,*d_selfa;
    double *d_nearr,*d_nearw;
    double *d_aeq;

    cuDoubleComplex* d_Z;

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

    CUDA_OK(cudaMalloc(&d_selfr, sz_selfr));
    CUDA_OK(cudaMalloc(&d_selfw, sz_selfw));
    CUDA_OK(cudaMalloc(&d_selfa, sz_selfa));

    CUDA_OK(cudaMalloc(&d_nearr, sz_nearr));
    CUDA_OK(cudaMalloc(&d_nearw, sz_nearw));

    CUDA_OK(cudaMalloc(&d_aeq, sz_aeq));

    CUDA_OK(cudaMemcpy(d_rreg, r_reg, sz_rreg, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_wreg, w_reg, sz_wreg, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_nearMat, nearMat, sz_nearMat, cudaMemcpyHostToDevice));

    CUDA_OK(cudaMemcpy(d_selfr, self_r, sz_selfr, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_selfw, self_w, sz_selfw, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_selfa, self_a, sz_selfa, cudaMemcpyHostToDevice));

    CUDA_OK(cudaMemcpy(d_nearr, near_r, sz_nearr, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_nearw, near_w, sz_nearw, cudaMemcpyHostToDevice));

    CUDA_OK(cudaMemcpy(d_aeq, aeqTri, sz_aeq, cudaMemcpyHostToDevice));

    CUDA_OK(cudaMalloc(&d_Z, (size_t)Ne*(size_t)Ne*sizeof(cuDoubleComplex)));

    dim3 block(8,8);
    dim3 grid((unsigned)((Ne + block.x - 1)/block.x),
              (unsigned)((Ne + block.y - 1)/block.y));

    kernel_Z_fastquad<<<grid,block>>>(
        d_Z, (int)Ne,
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

    // ---- copy Z back
    cuDoubleComplex* Zh = (cuDoubleComplex*)mxMalloc((size_t)Ne*(size_t)Ne*sizeof(cuDoubleComplex));
    CUDA_OK(cudaMemcpy(Zh, d_Z, (size_t)Ne*(size_t)Ne*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));

    // outputs (old MATLAB safe complex)
    plhs[0] = mxCreateDoubleMatrix((mwSize)Ne, (mwSize)Ne, mxCOMPLEX);
    double* Zre = mxGetPr(plhs[0]);
    double* Zim = mxGetPi(plhs[0]);
    mwSize NN = (mwSize)Ne*(mwSize)Ne;
    for(mwSize i=0;i<NN;i++){
        Zre[i] = cuCreal(Zh[i]);
        Zim[i] = cuCimag(Zh[i]);
    }

    plhs[1] = mxCreateDoubleMatrix((mwSize)Ne, 1, mxCOMPLEX);
    double* Vre = mxGetPr(plhs[1]);
    double* Vim = mxGetPi(plhs[1]);
    for(mwSize e=0;e<Ne;e++){
        Vre[e] = Vrhs_r[e];
        Vim[e] = 0.0;
    }

    // cleanup
    CUDA_OK(cudaFree(d_Z));

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

    mxFree(Zh);
    mxFree(Vrhs_r);
    mxFree(rpA);
    mxFree(rmA);
}
