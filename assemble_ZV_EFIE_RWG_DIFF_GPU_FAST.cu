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

// exp(-j*kR) with k = kr + j*ki
__device__ static inline cuDoubleComplex exp_negj_kR(cuDoubleComplex k, double R){
    double kr = cuCreal(k);
    double ki = cuCimag(k);
    double amp = exp(ki * R);
    double ph  = kr * R;
    return make_cuDoubleComplex(amp*cos(ph), -amp*sin(ph));
}
__device__ static inline cuDoubleComplex green(cuDoubleComplex k, double R){
    double inv = 1.0/(4.0*PI_D*R);
    return cmul_real(exp_negj_kR(k,R), inv);
}

// ---- RWG eval at point r on triangle t (t == plusTri[m] or minusTri[m]) ----
__device__ static inline void rwg_eval_point(
    int m, int t, d3 r,
    const int* plusTri, const int* minusTri,
    const int* plusSign, const int* minusSign,
    const double* len, const double* Ap, const double* Am,
    const d3* rp, const d3* rm,
    d3* f, double* divf)
{
    int tp = plusTri[m];
    int tm = minusTri[m];
    double l = len[m];

    if(t == tp && tp>0){
        int s = plusSign[m];
        double A = Ap[m];
        double scale = (double)s * (l/(2.0*A));
        d3 r0 = rp[m];
        *f = make_d3(scale*(r.x - r0.x), scale*(r.y - r0.y), scale*(r.z - r0.z));
        *divf = (double)s * (l/A);
        return;
    }
    if(t == tm && tm>0){
        int s = minusSign[m];
        double A = Am[m];
        double scale = (double)s * (l/(2.0*A));
        d3 r0 = rm[m];
        *f = make_d3(scale*(r0.x - r.x), scale*(r0.y - r.y), scale*(r0.z - r.z));
        *divf = (double)s * (-(l/A));
        return;
    }
    *f = make_d3(0,0,0);
    *divf = 0.0;
}

// ===============================
// KERNEL: Z entry compute (full fast: regular/near/self)
// Z is column-major (MATLAB style): Z[m + n*Ne]
// ===============================
__global__ void kernel_Z_fast(
    cuDoubleComplex* Z, int Ne,
    // RWG
    const int* plusTri, const int* minusTri,
    const int* plusSign, const int* minusSign,
    const double* len, const double* Ap, const double* Am,
    const d3* rp, const d3* rm,
    // prepack regular
    const d3* r_reg, const double* w_reg, int Nq, int Nt,
    // prepack nearMat
    const uint8_t* nearMat,
    // prepack aeqTri
    const double* aeqTri,
    // prepack self caches
    const d3* self_r, const double* self_w, const double* self_aSoft,
    int Nq2, int NsSelf,
    // prepack near caches
    const d3* near_r, const double* near_w, int NsNear,
    double near_alpha,
    // physics
    cuDoubleComplex k,
    cuDoubleComplex alpha, // j*omega*mu
    cuDoubleComplex beta   // 1/(j*omega*eps)
){
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    int m = blockIdx.y*blockDim.y + threadIdx.y;
    if(m>=Ne || n>=Ne) return;

    // triangles of RWG m,n (1-based in MATLAB). Here arrays are int32 already.
    int tm[2] = { plusTri[m], minusTri[m] };
    int tn[2] = { plusTri[n], minusTri[n] };

    cuDoubleComplex Zmn = make_cuDoubleComplex(0.0,0.0);

    for(int im=0; im<2; ++im){
        int t1 = tm[im];
        if(t1<=0) continue;
        for(int in=0; in<2; ++in){
            int t2 = tn[in];
            if(t2<=0) continue;

            // Decide type
            bool isSelf = (t1==t2);
            bool isNear = (!isSelf) && (nearMat[(t1-1) + (t2-1)*Nt] != 0);

            cuDoubleComplex Ivec = make_cuDoubleComplex(0.0,0.0);
            cuDoubleComplex Isca = make_cuDoubleComplex(0.0,0.0);

            if(!isSelf && !isNear){
                // -------- REGULAR FAST --------
                // r_reg: [Nq x Nt], w_reg: [Nq x Nt]
                for(int i=0;i<Nq;++i){
                    d3 ri = r_reg[i + (t1-1)*Nq];
                    double wi = w_reg[i + (t1-1)*Nq];
                    d3 fm; double divm;
                    rwg_eval_point(m, t1, ri, plusTri, minusTri, plusSign, minusSign, len, Ap, Am, rp, rm, &fm, &divm);

                    for(int j=0;j<Nq;++j){
                        d3 rj = r_reg[j + (t2-1)*Nq];
                        double wj = w_reg[j + (t2-1)*Nq];
                        d3 fn; double divn;
                        rwg_eval_point(n, t2, rj, plusTri, minusTri, plusSign, minusSign, len, Ap, Am, rp, rm, &fn, &divn);

                        d3 d = sub3(ri, rj);
                        double R = norm3(d);
                        if(R<1e-15) R=1e-15;

                        cuDoubleComplex G = green(k, R);

                        double dotff = dot3(fm, fn);
                        double divprod = divm*divn;
                        double W = wi*wj;

                        Ivec = cadd(Ivec, cmul_real(G, W*dotff));
                        Isca = cadd(Isca, cmul_real(G, W*divprod));
                    }
                }
            }
            else if(isNear){
                // -------- NEAR CACHED --------
                // near_r: [Nq2 x NsNear x Nt] packed as ((i + p*Nq2) + (t-1)*Nq2*NsNear)
                // near_w: [Nq2 x NsNear x Nt] same indexing (scalar weights)
                double aeq = fmin(aeqTri[t1-1], aeqTri[t2-1]);
                double a_soft = near_alpha * aeq;

                for(int p=0;p<NsNear;++p){
                    // points of subtri p on triangle t1
                    for(int i=0;i<Nq2;++i){
                        int idx1 = (i + p*Nq2) + (t1-1)*Nq2*NsNear;
                        d3 ri = near_r[idx1];
                        double wi = near_w[idx1];

                        d3 fm; double divm;
                        rwg_eval_point(m, t1, ri, plusTri, minusTri, plusSign, minusSign, len, Ap, Am, rp, rm, &fm, &divm);

                        for(int q=0;q<NsNear;++q){
                            for(int j=0;j<Nq2;++j){
                                int idx2 = (j + q*Nq2) + (t2-1)*Nq2*NsNear;
                                d3 rj = near_r[idx2];
                                double wj = near_w[idx2];

                                d3 fn; double divn;
                                rwg_eval_point(n, t2, rj, plusTri, minusTri, plusSign, minusSign, len, Ap, Am, rp, rm, &fn, &divn);

                                d3 d = sub3(ri, rj);
                                double R = norm3(d);
                                // softening
                                R = sqrt(R*R + a_soft*a_soft);
                                if(R<1e-15) R=1e-15;

                                cuDoubleComplex G = green(k, R);

                                double dotff = dot3(fm, fn);
                                double divprod = divm*divn;
                                double W = wi*wj;

                                Ivec = cadd(Ivec, cmul_real(G, W*dotff));
                                Isca = cadd(Isca, cmul_real(G, W*divprod));
                            }
                        }
                    }
                }
            }else{
                // -------- SELF CACHED --------
                // self_r: [Nq2 x NsSelf x Nt], self_w same, self_aSoft: [NsSelf x Nt]
                for(int p=0;p<NsSelf;++p){
                    double aS = self_aSoft[p + (t1-1)*NsSelf];

                    for(int i=0;i<Nq2;++i){
                        int idx1 = (i + p*Nq2) + (t1-1)*Nq2*NsSelf;
                        d3 ri = self_r[idx1];
                        double wi = self_w[idx1];

                        d3 fm; double divm;
                        rwg_eval_point(m, t1, ri, plusTri, minusTri, plusSign, minusSign, len, Ap, Am, rp, rm, &fm, &divm);

                        for(int q=0;q<NsSelf;++q){
                            for(int j=0;j<Nq2;++j){
                                int idx2 = (j + q*Nq2) + (t1-1)*Nq2*NsSelf;
                                d3 rj = self_r[idx2];
                                double wj = self_w[idx2];

                                d3 fn; double divn;
                                rwg_eval_point(n, t1, rj, plusTri, minusTri, plusSign, minusSign, len, Ap, Am, rp, rm, &fn, &divn);

                                d3 d = sub3(ri, rj);
                                double R = norm3(d);
                                if(p==q){
                                    R = sqrt(R*R + aS*aS);
                                }
                                if(R<1e-15) R=1e-15;

                                cuDoubleComplex G = green(k, R);

                                double dotff = dot3(fm, fn);
                                double divprod = divm*divn;
                                double W = wi*wj;

                                Ivec = cadd(Ivec, cmul_real(G, W*dotff));
                                Isca = cadd(Isca, cmul_real(G, W*divprod));
                            }
                        }
                    }
                }
            }

            // Z contribution: alpha*Ivec + beta*Isca
            Zmn = cadd(Zmn, cadd(cmul(alpha, Ivec), cmul(beta, Isca)));
        }
    }

    Z[m + n*Ne] = Zmn;
}

static void check(bool cond, const char* id, const char* msg){
    if(!cond) mexErrMsgIdAndTxt(id, msg);
}

// -------- MATLAB complex read helper --------
static void getComplexScalar(const mxArray* a, double* re, double* im){
#if MX_HAS_INTERLEAVED_COMPLEX
    if(mxIsComplex(a)){
        const mxComplexDouble* z = mxGetComplexDoubles(a);
        *re = z[0].real; *im = z[0].imag;
    }else{
        *re = mxGetScalar(a); *im = 0.0;
    }
#else
    *re = mxGetScalar(a);
    *im = mxIsComplex(a) ? *mxGetPi(a) : 0.0;
#endif
}

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]){
    // [Z, Vrhs] = assemble_ZV_EFIE_RWG_DIFF_GPU_FAST(rwg, prepack, k, omega, mu, eps, exc, opts)

    check(nrhs==8, "mex:args", "Expected 8 inputs: (rwg, prepack, k, omega, mu, eps, exc, opts)");
    check(nlhs==2, "mex:args", "Expected 2 outputs: [Z, Vrhs]");

    const mxArray* rwgS = prhs[0];
    const mxArray* preS = prhs[1];

    // k complex scalar
    double k_re,k_im; getComplexScalar(prhs[2], &k_re,&k_im);
    double omega = mxGetScalar(prhs[3]);
    double mu    = mxGetScalar(prhs[4]);
    double eps_re, eps_im; getComplexScalar(prhs[5], &eps_re,&eps_im);

    const mxArray* excS = prhs[6];
    const mxArray* optS = prhs[7];

    // --- read opts ---
    auto getOpt = [&](const char* name, double def)->double{
        mxArray* f = mxGetField(optS,0,name);
        return f ? mxGetScalar(f) : def;
    };
    double near_alpha = getOpt("near_alpha", 0.15);

    // --- read rwg fields ---
    auto F = [&](const char* name)->mxArray*{ return mxGetField(rwgS,0,name); };

    mxArray* plusTriMx  = F("plusTri");
    mxArray* minusTriMx = F("minusTri");
    mxArray* plusSignMx = F("plusSign");
    mxArray* minusSignMx= F("minusSign");
    mxArray* lenMx      = F("len");
    mxArray* ApMx       = F("Ap");
    mxArray* AmMx       = F("Am");
    mxArray* rpMx       = F("rp");
    mxArray* rmMx       = F("rm");
    mxArray* centerMx   = F("center");
    mxArray* zSignMx    = F("zSign");

    check(plusTriMx && minusTriMx && plusSignMx && minusSignMx, "mex:rwg","Missing tri/sign fields");
    check(mxIsInt32(plusTriMx) && mxIsInt32(minusTriMx), "mex:rwg","plusTri/minusTri must be int32");
    check(mxIsInt32(plusSignMx) && mxIsInt32(minusSignMx), "mex:rwg","plusSign/minusSign must be int32");

    int Ne = (int)mxGetNumberOfElements(plusTriMx);

    const int* plusTri  = (const int*)mxGetData(plusTriMx);
    const int* minusTri = (const int*)mxGetData(minusTriMx);
    const int* plusSign = (const int*)mxGetData(plusSignMx);
    const int* minusSign= (const int*)mxGetData(minusSignMx);

    const double* lenH = mxGetDoubles(lenMx);
    const double* ApH  = mxGetDoubles(ApMx);
    const double* AmH  = mxGetDoubles(AmMx);

    const double* rpH = mxGetDoubles(rpMx);
    const double* rmH = mxGetDoubles(rmMx);
    const double* cH  = mxGetDoubles(centerMx);
    const double* zH  = mxGetDoubles(zSignMx);

    check(mxIsDouble(rpMx) && mxGetN(rpMx)==3 && mxGetM(rpMx)==Ne, "mex:rwg","rp must be Ne x 3");
    check(mxIsDouble(rmMx) && mxGetN(rmMx)==3 && mxGetM(rmMx)==Ne, "mex:rwg","rm must be Ne x 3");

    // --- prepack fields ---
    auto P = [&](const char* name)->mxArray*{ return mxGetField(preS,0,name); };
    mxArray* nearMatMx = P("nearMat");
    mxArray* rregMx    = P("r_reg");
    mxArray* wregMx    = P("w_reg");
    mxArray* aeqMx     = P("aeqTri");
    mxArray* self_rMx  = P("self_r");
    mxArray* self_wMx  = P("self_w");
    mxArray* self_aMx  = P("self_aSoft");
    mxArray* near_rMx  = P("near_r");
    mxArray* near_wMx  = P("near_w");

    check(nearMatMx && rregMx && wregMx && aeqMx, "mex:pre","Missing prepack basic fields");
    check(mxIsUint8(nearMatMx), "mex:pre","nearMat must be uint8");
    check(mxIsDouble(rregMx) && mxIsDouble(wregMx), "mex:pre","r_reg/w_reg must be double");

    // sizes
    // r_reg: Nq x 3 x Nt
    const mwSize* dims_rreg = mxGetDimensions(rregMx);
    int Nq = (int)dims_rreg[0];
    int Nt = (int)dims_rreg[2];
    check((int)dims_rreg[1]==3, "mex:pre","r_reg second dim must be 3");

    // w_reg: Nq x Nt
    check((int)mxGetM(wregMx)==Nq, "mex:pre","w_reg rows must match Nq");
    check((int)mxGetN(wregMx)==Nt, "mex:pre","w_reg cols must be Nt");

    // self/near caches required (for full equivalence)
    check(self_rMx && self_wMx && self_aMx && near_rMx && near_wMx, "mex:pre",
          "Missing self/near cached fields. Use efie_prepack_for_mex(pre).");

    const mwSize* dims_self_r = mxGetDimensions(self_rMx); // Nq2 x 3 x NsSelf x Nt
    int Nq2    = (int)dims_self_r[0];
    int NsSelf = (int)dims_self_r[2];
    check((int)dims_self_r[1]==3 && (int)dims_self_r[3]==Nt, "mex:pre","self_r dims mismatch");

    const mwSize* dims_near_r = mxGetDimensions(near_rMx); // Nq2 x 3 x NsNear x Nt
    int NsNear = (int)dims_near_r[2];
    check((int)dims_near_r[0]==Nq2 && (int)dims_near_r[1]==3 && (int)dims_near_r[3]==Nt, "mex:pre","near_r dims mismatch");

    // pointers (host)
    const uint8_t* nearMatH = (const uint8_t*)mxGetData(nearMatMx);
    const double* rregH = mxGetDoubles(rregMx);
    const double* wregH = mxGetDoubles(wregMx);
    const double* aeqH  = mxGetDoubles(aeqMx);

    const double* self_rH = mxGetDoubles(self_rMx);
    const double* self_wH = mxGetDoubles(self_wMx);
    const double* self_aH = mxGetDoubles(self_aMx);

    const double* near_rH = mxGetDoubles(near_rMx);
    const double* near_wH = mxGetDoubles(near_wMx);

    // --- device allocate ---
    int *d_plusTri,*d_minusTri,*d_plusSign,*d_minusSign;
    double *d_len,*d_Ap,*d_Am;
    d3 *d_rp,*d_rm;

    CUDA_OK(cudaMalloc(&d_plusTri,  Ne*sizeof(int)));
    CUDA_OK(cudaMalloc(&d_minusTri, Ne*sizeof(int)));
    CUDA_OK(cudaMalloc(&d_plusSign, Ne*sizeof(int)));
    CUDA_OK(cudaMalloc(&d_minusSign,Ne*sizeof(int)));
    CUDA_OK(cudaMalloc(&d_len, Ne*sizeof(double)));
    CUDA_OK(cudaMalloc(&d_Ap,  Ne*sizeof(double)));
    CUDA_OK(cudaMalloc(&d_Am,  Ne*sizeof(double)));
    CUDA_OK(cudaMalloc(&d_rp,  Ne*sizeof(d3)));
    CUDA_OK(cudaMalloc(&d_rm,  Ne*sizeof(d3)));

    CUDA_OK(cudaMemcpy(d_plusTri, plusTri, Ne*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_minusTri,minusTri,Ne*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_plusSign,plusSign,Ne*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_minusSign,minusSign,Ne*sizeof(int),cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_len, lenH, Ne*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_Ap,  ApH,  Ne*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_Am,  AmH,  Ne*sizeof(double), cudaMemcpyHostToDevice));

    // pack rp/rm to d3
    d3* rpP = (d3*)mxMalloc(Ne*sizeof(d3));
    d3* rmP = (d3*)mxMalloc(Ne*sizeof(d3));
    for(int e=0;e<Ne;++e){
        rpP[e] = make_d3(rpH[e + 0*Ne], rpH[e + 1*Ne], rpH[e + 2*Ne]);
        rmP[e] = make_d3(rmH[e + 0*Ne], rmH[e + 1*Ne], rmH[e + 2*Ne]);
    }
    CUDA_OK(cudaMemcpy(d_rp, rpP, Ne*sizeof(d3), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_rm, rmP, Ne*sizeof(d3), cudaMemcpyHostToDevice));
    mxFree(rpP); mxFree(rmP);

    // prepack device
    uint8_t* d_nearMat;
    d3* d_rreg;
    double* d_wreg;
    double* d_aeq;
    d3* d_self_r;
    double* d_self_w;
    double* d_self_a;
    d3* d_near_r;
    double* d_near_w;

    CUDA_OK(cudaMalloc(&d_nearMat, (size_t)Nt*(size_t)Nt*sizeof(uint8_t)));
    CUDA_OK(cudaMemcpy(d_nearMat, nearMatH, (size_t)Nt*(size_t)Nt*sizeof(uint8_t), cudaMemcpyHostToDevice));

    // r_reg: Nq x 3 x Nt  (MATLAB column-major)
    // We'll pack to d3 array of size Nq*Nt: r_reg(i,t)
    // rregH layout: [i + Nq*comp + Nq*3*t]
    d3* rregP = (d3*)mxMalloc((size_t)Nq*(size_t)Nt*sizeof(d3));
    for(int t=0;t<Nt;++t){
        for(int i=0;i<Nq;++i){
            double x = rregH[i + Nq*0 + Nq*3*t];
            double y = rregH[i + Nq*1 + Nq*3*t];
            double z = rregH[i + Nq*2 + Nq*3*t];
            rregP[i + t*Nq] = make_d3(x,y,z);
        }
    }
    CUDA_OK(cudaMalloc(&d_rreg, (size_t)Nq*(size_t)Nt*sizeof(d3)));
    CUDA_OK(cudaMemcpy(d_rreg, rregP, (size_t)Nq*(size_t)Nt*sizeof(d3), cudaMemcpyHostToDevice));
    mxFree(rregP);

    CUDA_OK(cudaMalloc(&d_wreg, (size_t)Nq*(size_t)Nt*sizeof(double)));
    CUDA_OK(cudaMemcpy(d_wreg, wregH, (size_t)Nq*(size_t)Nt*sizeof(double), cudaMemcpyHostToDevice));

    CUDA_OK(cudaMalloc(&d_aeq, Nt*sizeof(double)));
    CUDA_OK(cudaMemcpy(d_aeq, aeqH, Nt*sizeof(double), cudaMemcpyHostToDevice));

    // self cache: self_r [Nq2 x 3 x NsSelf x Nt] -> d3 [Nq2*NsSelf*Nt]
    d3* selfRP = (d3*)mxMalloc((size_t)Nq2*(size_t)NsSelf*(size_t)Nt*sizeof(d3));
    for(int t=0;t<Nt;++t){
        for(int p=0;p<NsSelf;++p){
            for(int i=0;i<Nq2;++i){
                // MATLAB indexing: i + Nq2*comp + Nq2*3*p + Nq2*3*NsSelf*t
                size_t base = (size_t)i + (size_t)Nq2*0 + (size_t)Nq2*3*(size_t)p + (size_t)Nq2*3*(size_t)NsSelf*(size_t)t;
                double x = self_rH[ base ];
                double y = self_rH[ base + (size_t)Nq2 ];
                double z = self_rH[ base + (size_t)2*(size_t)Nq2 ];
                selfRP[(size_t)i + (size_t)p*(size_t)Nq2 + (size_t)t*(size_t)Nq2*(size_t)NsSelf] = make_d3(x,y,z);
            }
        }
    }
    CUDA_OK(cudaMalloc(&d_self_r, (size_t)Nq2*(size_t)NsSelf*(size_t)Nt*sizeof(d3)));
    CUDA_OK(cudaMemcpy(d_self_r, selfRP, (size_t)Nq2*(size_t)NsSelf*(size_t)Nt*sizeof(d3), cudaMemcpyHostToDevice));
    mxFree(selfRP);

    CUDA_OK(cudaMalloc(&d_self_w, (size_t)Nq2*(size_t)NsSelf*(size_t)Nt*sizeof(double)));
    CUDA_OK(cudaMemcpy(d_self_w, self_wH, (size_t)Nq2*(size_t)NsSelf*(size_t)Nt*sizeof(double), cudaMemcpyHostToDevice));

    CUDA_OK(cudaMalloc(&d_self_a, (size_t)NsSelf*(size_t)Nt*sizeof(double)));
    CUDA_OK(cudaMemcpy(d_self_a, self_aH, (size_t)NsSelf*(size_t)Nt*sizeof(double), cudaMemcpyHostToDevice));

    // near cache: near_r [Nq2 x 3 x NsNear x Nt] -> d3 [Nq2*NsNear*Nt]
    d3* nearRP = (d3*)mxMalloc((size_t)Nq2*(size_t)NsNear*(size_t)Nt*sizeof(d3));
    for(int t=0;t<Nt;++t){
        for(int p=0;p<NsNear;++p){
            for(int i=0;i<Nq2;++i){
                size_t base = (size_t)i + (size_t)Nq2*0 + (size_t)Nq2*3*(size_t)p + (size_t)Nq2*3*(size_t)NsNear*(size_t)t;
                double x = near_rH[ base ];
                double y = near_rH[ base + (size_t)Nq2 ];
                double z = near_rH[ base + (size_t)2*(size_t)Nq2 ];
                nearRP[(size_t)i + (size_t)p*(size_t)Nq2 + (size_t)t*(size_t)Nq2*(size_t)NsNear] = make_d3(x,y,z);
            }
        }
    }
    CUDA_OK(cudaMalloc(&d_near_r, (size_t)Nq2*(size_t)NsNear*(size_t)Nt*sizeof(d3)));
    CUDA_OK(cudaMemcpy(d_near_r, nearRP, (size_t)Nq2*(size_t)NsNear*(size_t)Nt*sizeof(d3), cudaMemcpyHostToDevice));
    mxFree(nearRP);

    CUDA_OK(cudaMalloc(&d_near_w, (size_t)Nq2*(size_t)NsNear*(size_t)Nt*sizeof(double)));
    CUDA_OK(cudaMemcpy(d_near_w, near_wH, (size_t)Nq2*(size_t)NsNear*(size_t)Nt*sizeof(double), cudaMemcpyHostToDevice));

    // --- output Z (complex Ne x Ne) ---
    plhs[0] = mxCreateDoubleMatrix(Ne, Ne, mxCOMPLEX);
#if MX_HAS_INTERLEAVED_COMPLEX
    mxComplexDouble* Zout = mxGetComplexDoubles(plhs[0]);
#else
    double* Zre = mxGetPr(plhs[0]);
    double* Zim = mxGetPi(plhs[0]);
#endif

    cuDoubleComplex* d_Z;
    CUDA_OK(cudaMalloc(&d_Z, (size_t)Ne*(size_t)Ne*sizeof(cuDoubleComplex)));

    // physics: alpha = j*omega*mu, beta = 1/(j*omega*eps)
    // eps complex
    // beta = 1/(j*omega*(eps_re+j*eps_im))  (do complex division on host)
    double ar=0, ai=omega*mu; // j*omega*mu
    cuDoubleComplex alpha = make_cuDoubleComplex(ar, ai);

    // beta host
    // denom = j*omega*eps = j*omega*(e_re + j e_im) = j*omega*e_re + j*omega*j*e_im = ( -omega*e_im ) + j*(omega*e_re)
    double denom_re = -omega*eps_im;
    double denom_im =  omega*eps_re;
    double denom_norm = denom_re*denom_re + denom_im*denom_im;
    double beta_re =  denom_re/denom_norm;
    double beta_im = -denom_im/denom_norm; // 1/(a+jb) = (a-jb)/(a^2+b^2)
    cuDoubleComplex beta = make_cuDoubleComplex(beta_re, beta_im);

    cuDoubleComplex k_c = make_cuDoubleComplex(k_re, k_im);

    dim3 block(16,16);
    dim3 grid((Ne + block.x - 1)/block.x, (Ne + block.y - 1)/block.y);

    kernel_Z_fast<<<grid,block>>>(
        d_Z, Ne,
        d_plusTri,d_minusTri,d_plusSign,d_minusSign,
        d_len,d_Ap,d_Am,d_rp,d_rm,
        d_rreg,d_wreg,Nq,Nt,
        d_nearMat,
        d_aeq,
        d_self_r,d_self_w,d_self_a,Nq2,NsSelf,
        d_near_r,d_near_w,NsNear,
        near_alpha,
        k_c, alpha, beta
    );
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaDeviceSynchronize());

    // copy Z back
    cuDoubleComplex* Zh = (cuDoubleComplex*)mxMalloc((size_t)Ne*(size_t)Ne*sizeof(cuDoubleComplex));
    CUDA_OK(cudaMemcpy(Zh, d_Z, (size_t)Ne*(size_t)Ne*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));

#if MX_HAS_INTERLEAVED_COMPLEX
    for(int j=0;j<Ne*Ne;++j){
        Zout[j].real = cuCreal(Zh[j]);
        Zout[j].imag = cuCimag(Zh[j]);
    }
#else
    for(int j=0;j<Ne*Ne;++j){
        Zre[j] = cuCreal(Zh[j]);
        Zim[j] = cuCimag(Zh[j]);
    }
#endif
    mxFree(Zh);

    // --- build Vrhs on HOST (same as rhs_edge_strip_diff_gaussian) ---
    mxArray* patchEdgeMx = mxGetField(excS,0,"patchEdge");
    mxArray* groundEdgeMx= mxGetField(excS,0,"groundEdge");
    mxArray* V0Mx        = mxGetField(excS,0,"V0");
    mxArray* spreadMx    = mxGetField(excS,0,"spreadRadius");
    check(patchEdgeMx && groundEdgeMx && V0Mx && spreadMx, "mex:exc","exc must have patchEdge, groundEdge, V0, spreadRadius");

    int patchEdge = (int)mxGetScalar(patchEdgeMx);
    int groundEdge= (int)mxGetScalar(groundEdgeMx);
    double V0 = mxGetScalar(V0Mx);
    double spread = mxGetScalar(spreadMx);

    plhs[1] = mxCreateDoubleMatrix(Ne, 1, mxCOMPLEX);
#if MX_HAS_INTERLEAVED_COMPLEX
    mxComplexDouble* Vout = mxGetComplexDoubles(plhs[1]);
#else
    double* Vre = mxGetPr(plhs[1]);
    double* Vim = mxGetPi(plhs[1]);
#endif

    // compute gaussian weights on layer using rwg.center & rwg.zSign
    auto local_gauss = [&](int edge0_1based, double amp, double* out){
        int e0 = edge0_1based-1;
        int layer0 = (zH[e0] > 0) ? +1 : -1;
        double r0x = cH[e0 + 0*Ne], r0y = cH[e0 + 1*Ne], r0z = cH[e0 + 2*Ne];
        double sumw=0.0;
        for(int e=0;e<Ne;++e){
            int lay = (zH[e] > 0) ? +1 : -1;
            if(lay!=layer0){ out[e]=0.0; continue; }
            double dx = cH[e + 0*Ne]-r0x;
            double dy = cH[e + 1*Ne]-r0y;
            double dz = cH[e + 2*Ne]-r0z;
            double d = sqrt(dx*dx+dy*dy+dz*dz);
            double w = exp(-(d/spread)*(d/spread));
            out[e]=w; sumw += w;
        }
        if(sumw < 1e-30) mexErrMsgIdAndTxt("mex:rhs","spreadRadius too small (no weights)");
        for(int e=0;e<Ne;++e) out[e] = amp*(out[e]/sumw);
    };

    double* tmp = (double*)mxMalloc(Ne*sizeof(double));
    double* Vr  = (double*)mxMalloc(Ne*sizeof(double));
    for(int e=0;e<Ne;++e) Vr[e]=0.0;

    local_gauss(patchEdge,  +V0/2.0, tmp);
    for(int e=0;e<Ne;++e) Vr[e]+=tmp[e];
    local_gauss(groundEdge, -V0/2.0, tmp);
    for(int e=0;e<Ne;++e) Vr[e]+=tmp[e];

#if MX_HAS_INTERLEAVED_COMPLEX
    for(int e=0;e<Ne;++e){ Vout[e].real = Vr[e]; Vout[e].imag = 0.0; }
#else
    for(int e=0;e<Ne;++e){ Vre[e] = Vr[e]; Vim[e] = 0.0; }
#endif

    mxFree(tmp); mxFree(Vr);

    // cleanup device
    CUDA_OK(cudaFree(d_Z));

    CUDA_OK(cudaFree(d_plusTri)); CUDA_OK(cudaFree(d_minusTri));
    CUDA_OK(cudaFree(d_plusSign));CUDA_OK(cudaFree(d_minusSign));
    CUDA_OK(cudaFree(d_len)); CUDA_OK(cudaFree(d_Ap)); CUDA_OK(cudaFree(d_Am));
    CUDA_OK(cudaFree(d_rp));  CUDA_OK(cudaFree(d_rm));

    CUDA_OK(cudaFree(d_nearMat));
    CUDA_OK(cudaFree(d_rreg)); CUDA_OK(cudaFree(d_wreg));
    CUDA_OK(cudaFree(d_aeq));
    CUDA_OK(cudaFree(d_self_r)); CUDA_OK(cudaFree(d_self_w)); CUDA_OK(cudaFree(d_self_a));
    CUDA_OK(cudaFree(d_near_r)); CUDA_OK(cudaFree(d_near_w));
}
