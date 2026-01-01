// patch_sweep_cuda.cu
// mexcuda -R2018a patch_sweep_cuda.cu -lcusolver -lcublas  NVCCFLAGS="$NVCCFLAGS -allow-unsupported-compiler"
//
// Inputs (20):
//  1  V        [Nv x 3] double
//  2  F        [Nt x 3] int32 (1-based)
//  3  edge     [Ne x 2] int32 (1-based)   rwg.edge
//  4  plusTri  [Ne x 1] int32 (1-based, 0 allowed)
//  5  minusTri [Ne x 1] int32
//  6  plusSign [Ne x 1] int32  (+1/-1)   rwg.plusSign
//  7  minusSign[Ne x 1] int32  (+1/-1)   rwg.minusSign
//  8  rp       [Ne x 3] double rwg.rp
//  9  rm       [Ne x 3] double rwg.rm
// 10  Ap       [Ne x 1] double rwg.Ap
// 11  Am       [Ne x 1] double rwg.Am
// 12  len      [Ne x 1] double rwg.len
// 13  freqs    [Nf x 1] double (Hz)
// 14  mu       scalar double (mu0)
// 15  eps      scalar complex double (eps_eff = eps0*epsr_eff*(1 - j*tand))
// 16  patchEdge  int32 scalar (1-based edge index in RWG list)
// 17  groundEdge int32 scalar
// 18  V0       scalar double
// 19  spreadRadius scalar double
// 20  Z0       scalar double (e.g., 50)
//
// Outputs (2):
//  [Zin, S11] both [Nf x 1] complex double
//
// Model:
//  Zmn = sum_{sa,sb in {plus,minus}} [ alpha * dot(fm_sa, fn_sb) * G * w_sa w_sb
//                                     + beta  * (divm_sa divn_sb) * G * w_sa w_sb ]
//  alpha = j*omega*mu
//  beta  = 1/(j*omega*eps)
//  G = exp(-j*k*R)/(4*pi*R), k = omega*sqrt(mu*eps)
//  R softened only when triangle indices equal.

#include "mex.h"
#include "matrix.h"

#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <cuComplex.h>

#include <complex>
#include <math.h>

#ifndef PI_D
#define PI_D 3.141592653589793238462643383279502884
#endif

#define CUDA_OK(call) do{ cudaError_t e=(call); if(e!=cudaSuccess){ mexErrMsgIdAndTxt("cuda:rt", cudaGetErrorString(e)); } }while(0)
#define CUSOLVER_OK(call) do{ cusolverStatus_t s=(call); if(s!=CUSOLVER_STATUS_SUCCESS){ mexErrMsgIdAndTxt("cuda:cusolver","cuSOLVER error"); } }while(0)
#define CUBLAS_OK(call) do{ cublasStatus_t s=(call); if(s!=CUBLAS_STATUS_SUCCESS){ mexErrMsgIdAndTxt("cuda:cublas","cuBLAS error"); } }while(0)

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

// exp(-j*(kR)) where k complex = kr + j*ki
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

// Kernel: centroid 1-pt assembly
// pre arrays:
//  posSide[2][Ne], fSide[2][Ne], divSide[2][Ne], wSide[2][Ne], tSide[2][Ne]
//  asoftTri[Nt]
__global__ void kernel_Z_centroid(cuDoubleComplex* Z, int Ne,
                                  const d3* pos1, const d3* pos2,
                                  const d3* f1,   const d3* f2,
                                  const double* div1, const double* div2,
                                  const double* w1,   const double* w2,
                                  const int* t1,      const int* t2,
                                  const double* asoftTri,
                                  cuDoubleComplex k,
                                  cuDoubleComplex alpha,
                                  cuDoubleComplex beta)
{
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if(m>=Ne || n>=Ne) return;

    cuDoubleComplex Zmn = make_cuDoubleComplex(0.0,0.0);

    // side arrays via pointers
    // sa=0 -> plus(1), sa=1 -> minus(2)
    for(int sa=0; sa<2; ++sa){
        d3 pm = (sa==0) ? pos1[m] : pos2[m];
        d3 fm = (sa==0) ? f1[m]   : f2[m];
        double dvm = (sa==0) ? div1[m] : div2[m];
        double wm  = (sa==0) ? w1[m]   : w2[m];
        int tm      = (sa==0) ? t1[m]   : t2[m];
        if(wm==0.0 || tm==0) continue;

        for(int sb=0; sb<2; ++sb){
            d3 pn = (sb==0) ? pos1[n] : pos2[n];
            d3 fn = (sb==0) ? f1[n]   : f2[n];
            double dvn = (sb==0) ? div1[n] : div2[n];
            double wn  = (sb==0) ? w1[n]   : w2[n];
            int tn      = (sb==0) ? t1[n]   : t2[n];
            if(wn==0.0 || tn==0) continue;

            d3 diff = sub3(pm, pn);
            double R = norm3(diff);

            // self-softening only if same triangle
            if(tm == tn){
                double a = asoftTri[tm-1];  // t is 1-based
                R = sqrt(R*R + a*a);
            }
            if(R < 1e-15) R = 1e-15;

            cuDoubleComplex G = green(k, R);

            double dotff = dot3(fm, fn);
            double divprod = dvm * dvn;
            double w = wm * wn;

            cuDoubleComplex term_vec = cmul_real(G, dotff * w);
            cuDoubleComplex term_sca = cmul_real(G, divprod * w);

            Zmn = cadd(Zmn, cadd(cmul(alpha, term_vec), cmul(beta, term_sca)));
        }
    }

    Z[m + n*Ne] = Zmn; // column-major
}

// -------- Host helpers --------
static std::complex<double> mx_to_cplx(const mxArray* a){
    if(!mxIsComplex(a)){
        return std::complex<double>(mxGetScalar(a), 0.0);
    }
#if MX_HAS_INTERLEAVED_COMPLEX
    const mxComplexDouble* z = mxGetComplexDoubles(a);
    return std::complex<double>(z[0].real, z[0].imag);
#else
    return std::complex<double>(mxGetScalar(a), *mxGetPi(a));
#endif
}

static void check(bool cond, const char* id, const char* msg){
    if(!cond) mexErrMsgIdAndTxt(id, msg);
}

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    check(nrhs==20, "mex:args", "Expected 20 inputs.");
    check(nlhs==2,  "mex:args", "Expected 2 outputs [Zin,S11].");

    // Inputs
    const mxArray* Vmx = prhs[0];
    const mxArray* Fmx = prhs[1];
    const mxArray* Emx = prhs[2];
    const mxArray* plusTriMx  = prhs[3];
    const mxArray* minusTriMx = prhs[4];
    const mxArray* plusSignMx = prhs[5];
    const mxArray* minusSignMx= prhs[6];
    const mxArray* rpMx = prhs[7];
    const mxArray* rmMx = prhs[8];
    const mxArray* ApMx = prhs[9];
    const mxArray* AmMx = prhs[10];
    const mxArray* lenMx= prhs[11];
    const mxArray* freqsMx = prhs[12];

    double mu = mxGetScalar(prhs[13]);
    std::complex<double> eps = mx_to_cplx(prhs[14]);

    int patchEdge  = (int)mxGetScalar(prhs[15]);
    int groundEdge = (int)mxGetScalar(prhs[16]);
    double V0 = mxGetScalar(prhs[17]);
    double spread = mxGetScalar(prhs[18]);
    double Z0 = mxGetScalar(prhs[19]);

    // Type checks
    check(mxIsDouble(Vmx) && mxGetN(Vmx)==3, "mex:V", "V must be [Nv x 3] double.");
    check(mxIsInt32(Fmx) && mxGetN(Fmx)==3, "mex:F", "F must be [Nt x 3] int32.");
    check(mxIsInt32(Emx) && mxGetN(Emx)==2, "mex:edge", "edge must be [Ne x 2] int32.");

    mwSize Nv = mxGetM(Vmx);
    mwSize Nt = mxGetM(Fmx);
    mwSize Ne = mxGetM(Emx);

    check(mxGetM(plusTriMx)==Ne && mxIsInt32(plusTriMx), "mex:plusTri", "plusTri must be int32 [Ne x 1].");
    check(mxGetM(minusTriMx)==Ne && mxIsInt32(minusTriMx),"mex:minusTri","minusTri must be int32 [Ne x 1].");
    check(mxGetM(plusSignMx)==Ne && mxIsInt32(plusSignMx),"mex:plusSign","plusSign must be int32 [Ne x 1].");
    check(mxGetM(minusSignMx)==Ne && mxIsInt32(minusSignMx),"mex:minusSign","minusSign must be int32 [Ne x 1].");

    check(mxIsDouble(rpMx) && mxGetM(rpMx)==Ne && mxGetN(rpMx)==3, "mex:rp", "rp must be [Ne x 3] double.");
    check(mxIsDouble(rmMx) && mxGetM(rmMx)==Ne && mxGetN(rmMx)==3, "mex:rm", "rm must be [Ne x 3] double.");
    check(mxIsDouble(ApMx) && mxGetNumberOfElements(ApMx)==Ne, "mex:Ap", "Ap must be [Ne x 1] double.");
    check(mxIsDouble(AmMx) && mxGetNumberOfElements(AmMx)==Ne, "mex:Am", "Am must be [Ne x 1] double.");
    check(mxIsDouble(lenMx)&& mxGetNumberOfElements(lenMx)==Ne,"mex:len","len must be [Ne x 1] double.");

    check(mxIsDouble(freqsMx), "mex:freqs", "freqs must be double vector.");
    mwSize Nf = mxGetNumberOfElements(freqsMx);

    check(patchEdge>=1 && patchEdge<=(int)Ne, "mex:port", "patchEdge out of range.");
    check(groundEdge>=1 && groundEdge<=(int)Ne, "mex:port", "groundEdge out of range.");

    const double* Vh = mxGetDoubles(Vmx);
    const int* Fh = (const int*)mxGetData(Fmx);
    const int* Eh = (const int*)mxGetData(Emx);

    const int* plusTri  = (const int*)mxGetData(plusTriMx);
    const int* minusTri = (const int*)mxGetData(minusTriMx);
    const int* plusSign = (const int*)mxGetData(plusSignMx);
    const int* minusSign= (const int*)mxGetData(minusSignMx);

    const double* rpH = mxGetDoubles(rpMx);
    const double* rmH = mxGetDoubles(rmMx);
    const double* ApH = mxGetDoubles(ApMx);
    const double* AmH = mxGetDoubles(AmMx);
    const double* lenH= mxGetDoubles(lenMx);

    const double* freqs = mxGetDoubles(freqsMx);

    // ---- Precompute triangle centroid & area & asoft (host)
    d3* triCent = (d3*)mxMalloc(Nt*sizeof(d3));
    double* triArea = (double*)mxMalloc(Nt*sizeof(double));
    double* asoftTri = (double*)mxMalloc(Nt*sizeof(double));

    for(mwSize t=0; t<Nt; ++t){
        int i1 = Fh[t + 0*Nt] - 1;
        int i2 = Fh[t + 1*Nt] - 1;
        int i3 = Fh[t + 2*Nt] - 1;
        check(i1>=0 && i2>=0 && i3>=0 && i1<(int)Nv && i2<(int)Nv && i3<(int)Nv, "mex:F", "F has invalid indices.");

        d3 r1 = make_d3(Vh[i1 + 0*Nv], Vh[i1 + 1*Nv], Vh[i1 + 2*Nv]);
        d3 r2 = make_d3(Vh[i2 + 0*Nv], Vh[i2 + 1*Nv], Vh[i2 + 2*Nv]);
        d3 r3 = make_d3(Vh[i3 + 0*Nv], Vh[i3 + 1*Nv], Vh[i3 + 2*Nv]);

        triCent[t] = make_d3((r1.x+r2.x+r3.x)/3.0, (r1.y+r2.y+r3.y)/3.0, (r1.z+r2.z+r3.z)/3.0);

        d3 e1 = sub3(r2,r1);
        d3 e2 = sub3(r3,r1);
        double cx = e1.y*e2.z - e1.z*e2.y;
        double cy = e1.z*e2.x - e1.x*e2.z;
        double cz = e1.x*e2.y - e1.y*e2.x;
        double area2 = sqrt(cx*cx + cy*cy + cz*cz); // 2A
        triArea[t] = 0.5 * area2;
        // self softening length a = 0.2*sqrt(A/pi) (sweep için sabit bir default; istersen input yaparız)
        asoftTri[t] = 0.20 * sqrt(triArea[t] / PI_D);
    }

    // ---- Precompute RWG side data (host)
    d3* pos1 = (d3*)mxMalloc(Ne*sizeof(d3));
    d3* pos2 = (d3*)mxMalloc(Ne*sizeof(d3));
    d3* f1   = (d3*)mxMalloc(Ne*sizeof(d3));
    d3* f2   = (d3*)mxMalloc(Ne*sizeof(d3));
    double* div1 = (double*)mxMalloc(Ne*sizeof(double));
    double* div2 = (double*)mxMalloc(Ne*sizeof(double));
    double* w1   = (double*)mxMalloc(Ne*sizeof(double));
    double* w2   = (double*)mxMalloc(Ne*sizeof(double));
    int* t1 = (int*)mxMalloc(Ne*sizeof(int));
    int* t2 = (int*)mxMalloc(Ne*sizeof(int));

    for(mwSize e=0; e<Ne; ++e){
        int tp = plusTri[e];
        int tm = minusTri[e];
        t1[e]=tp; t2[e]=tm;

        // positions at triangle centroids
        pos1[e] = (tp>0) ? triCent[tp-1] : make_d3(0,0,0);
        pos2[e] = (tm>0) ? triCent[tm-1] : make_d3(0,0,0);

        // weights
        w1[e] = (tp>0) ? ApH[e] : 0.0;
        w2[e] = (tm>0) ? AmH[e] : 0.0;

        double l = lenH[e];

        // rp/rm vectors
        d3 rp = make_d3(rpH[e + 0*Ne], rpH[e + 1*Ne], rpH[e + 2*Ne]);
        d3 rm = make_d3(rmH[e + 0*Ne], rmH[e + 1*Ne], rmH[e + 2*Ne]);

        // plus
        if(tp>0 && ApH[e]>0){
            int s = plusSign[e];
            double A = ApH[e];
            double scale = (double)s * (l/(2.0*A));
            d3 c = pos1[e];
            f1[e] = make_d3(scale*(c.x-rp.x), scale*(c.y-rp.y), scale*(c.z-rp.z));
            div1[e] = (double)s * (l/A);
        }else{
            f1[e]=make_d3(0,0,0); div1[e]=0.0;
        }

        // minus
        if(tm>0 && AmH[e]>0){
            int s = minusSign[e];
            double A = AmH[e];
            double scale = (double)s * (l/(2.0*A));
            d3 c = pos2[e];
            f2[e] = make_d3(scale*(rm.x-c.x), scale*(rm.y-c.y), scale*(rm.z-c.z));
            div2[e] = (double)s * (-(l/A));
        }else{
            f2[e]=make_d3(0,0,0); div2[e]=0.0;
        }
    }

    // ---- Build Vrhs once (host): differential gaussian edge-strip on same layer
    // edge centers + zSign
    d3* ecent = (d3*)mxMalloc(Ne*sizeof(d3));
    double zmin=1e100, zmax=-1e100;
    for(mwSize e=0; e<Ne; ++e){
        int a = Eh[e + 0*Ne] - 1;
        int b = Eh[e + 1*Ne] - 1;
        d3 ra = make_d3(Vh[a + 0*Nv], Vh[a + 1*Nv], Vh[a + 2*Nv]);
        d3 rb = make_d3(Vh[b + 0*Nv], Vh[b + 1*Nv], Vh[b + 2*Nv]);
        d3 c = make_d3(0.5*(ra.x+rb.x), 0.5*(ra.y+rb.y), 0.5*(ra.z+rb.z));
        ecent[e]=c;
        if(c.z<zmin) zmin=c.z;
        if(c.z>zmax) zmax=c.z;
    }
    double zmid = 0.5*(zmin+zmax);

    auto build_gauss = [&](int edge1based, double amp, double* out){
        int e0 = edge1based-1;
        int layer = (ecent[e0].z > zmid) ? +1 : -1;
        d3 r0 = ecent[e0];
        double sumw = 0.0;
        for(mwSize e=0; e<Ne; ++e){
            int lay = (ecent[e].z > zmid) ? +1 : -1;
            if(lay!=layer){ out[e]=0.0; continue; }
            d3 d = sub3(ecent[e], r0);
            double dist = norm3(d);
            double w = exp(-(dist/spread)*(dist/spread));
            out[e]=w;
            sumw += w;
        }
        check(sumw>1e-30, "mex:rhs", "spreadRadius too small (no weights).");
        for(mwSize e=0;e<Ne;++e) out[e] = amp*(out[e]/sumw);
    };

    double* rhs_r = (double*)mxMalloc(Ne*sizeof(double));
    double* tmp   = (double*)mxMalloc(Ne*sizeof(double));
    for(mwSize e=0;e<Ne;++e) rhs_r[e]=0.0;

    build_gauss(patchEdge,  +V0/2.0, tmp);
    for(mwSize e=0;e<Ne;++e) rhs_r[e]+=tmp[e];
    build_gauss(groundEdge, -V0/2.0, tmp);
    for(mwSize e=0;e<Ne;++e) rhs_r[e]+=tmp[e];

    // ---- Allocate device pre arrays
    d3 *d_pos1,*d_pos2,*d_f1,*d_f2;
    double *d_div1,*d_div2,*d_w1,*d_w2,*d_asoft;
    int *d_t1,*d_t2;

    CUDA_OK(cudaMalloc(&d_pos1, Ne*sizeof(d3)));
    CUDA_OK(cudaMalloc(&d_pos2, Ne*sizeof(d3)));
    CUDA_OK(cudaMalloc(&d_f1,   Ne*sizeof(d3)));
    CUDA_OK(cudaMalloc(&d_f2,   Ne*sizeof(d3)));
    CUDA_OK(cudaMalloc(&d_div1, Ne*sizeof(double)));
    CUDA_OK(cudaMalloc(&d_div2, Ne*sizeof(double)));
    CUDA_OK(cudaMalloc(&d_w1,   Ne*sizeof(double)));
    CUDA_OK(cudaMalloc(&d_w2,   Ne*sizeof(double)));
    CUDA_OK(cudaMalloc(&d_t1,   Ne*sizeof(int)));
    CUDA_OK(cudaMalloc(&d_t2,   Ne*sizeof(int)));
    CUDA_OK(cudaMalloc(&d_asoft, Nt*sizeof(double)));

    CUDA_OK(cudaMemcpy(d_pos1, pos1, Ne*sizeof(d3), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_pos2, pos2, Ne*sizeof(d3), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_f1,   f1,   Ne*sizeof(d3), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_f2,   f2,   Ne*sizeof(d3), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_div1, div1, Ne*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_div2, div2, Ne*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_w1,   w1,   Ne*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_w2,   w2,   Ne*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_t1,   t1,   Ne*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_t2,   t2,   Ne*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_asoft, asoftTri, Nt*sizeof(double), cudaMemcpyHostToDevice));

    // RHS device (cuDoubleComplex)
    cuDoubleComplex* d_b;
    CUDA_OK(cudaMalloc(&d_b, Ne*sizeof(cuDoubleComplex)));
    // build b as complex with imag=0
    cuDoubleComplex* b_h = (cuDoubleComplex*)mxMalloc(Ne*sizeof(cuDoubleComplex));
    for(mwSize i=0;i<Ne;++i) b_h[i] = make_cuDoubleComplex(rhs_r[i], 0.0);
    CUDA_OK(cudaMemcpy(d_b, b_h, Ne*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

    // Output arrays
    plhs[0] = mxCreateDoubleMatrix((mwSize)Nf, 1, mxCOMPLEX); // Zin
    plhs[1] = mxCreateDoubleMatrix((mwSize)Nf, 1, mxCOMPLEX); // S11

#if MX_HAS_INTERLEAVED_COMPLEX
    mxComplexDouble* ZinOut = mxGetComplexDoubles(plhs[0]);
    mxComplexDouble* S11Out = mxGetComplexDoubles(plhs[1]);
#else
    double* ZinRe = mxGetPr(plhs[0]); double* ZinIm = mxGetPi(plhs[0]);
    double* S11Re = mxGetPr(plhs[1]); double* S11Im = mxGetPi(plhs[1]);
#endif

    // cuSOLVER setup
    cusolverDnHandle_t solver;
    CUSOLVER_OK(cusolverDnCreate(&solver));

    // Allocate Z on device
    cuDoubleComplex* d_A;
    CUDA_OK(cudaMalloc(&d_A, (size_t)Ne*(size_t)Ne*sizeof(cuDoubleComplex)));

    // pivot, info
    int* d_ipiv; int* d_info;
    CUDA_OK(cudaMalloc(&d_ipiv, Ne*sizeof(int)));
    CUDA_OK(cudaMalloc(&d_info, sizeof(int)));

    int lwork=0;
    // We'll query lwork each iteration after Z is built (same size, so once is fine)
    cuDoubleComplex* d_work = nullptr;

    dim3 block(16,16);
    dim3 grid((unsigned)((Ne + block.x - 1)/block.x),
              (unsigned)((Ne + block.y - 1)/block.y));

    // Solve per frequency
    for(mwSize fi=0; fi<Nf; ++fi){
        double fHz = freqs[fi];
        double omega = 2.0*PI_D*fHz;

        std::complex<double> j(0.0,1.0);
        std::complex<double> k = omega * std::sqrt(std::complex<double>(mu,0.0)*eps);
        std::complex<double> alpha = j*omega*mu;
        std::complex<double> beta  = 1.0/(j*omega*eps);

        cuDoubleComplex k_c   = make_cuDoubleComplex(k.real(), k.imag());
        cuDoubleComplex a_c   = make_cuDoubleComplex(alpha.real(), alpha.imag());
        cuDoubleComplex b_cpl = make_cuDoubleComplex(beta.real(), beta.imag());

        // Assemble Z into d_A
        kernel_Z_centroid<<<grid,block>>>(d_A, (int)Ne,
                                          d_pos1,d_pos2,d_f1,d_f2,
                                          d_div1,d_div2,d_w1,d_w2,
                                          d_t1,d_t2,d_asoft,
                                          k_c, a_c, b_cpl);
        CUDA_OK(cudaGetLastError());
        CUDA_OK(cudaDeviceSynchronize());

        // Copy RHS to x (in-place solve needs B vector)
        // We'll reuse d_b as RHS each time, but getrf/getrs overwrites B.
        // So we make a copy:
        cuDoubleComplex* d_x;
        CUDA_OK(cudaMalloc(&d_x, Ne*sizeof(cuDoubleComplex)));
        CUDA_OK(cudaMemcpy(d_x, d_b, Ne*sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice));

        // Workspace query once
        if(d_work==nullptr){
            CUSOLVER_OK(cusolverDnZgetrf_bufferSize(solver, (int)Ne, (int)Ne, d_A, (int)Ne, &lwork));
            CUDA_OK(cudaMalloc(&d_work, (size_t)lwork*sizeof(cuDoubleComplex)));
        }

        // LU factorization in-place on d_A
        CUSOLVER_OK(cusolverDnZgetrf(solver, (int)Ne, (int)Ne, d_A, (int)Ne, d_work, d_ipiv, d_info));
        // Solve A x = b
        CUSOLVER_OK(cusolverDnZgetrs(solver, CUBLAS_OP_N, (int)Ne, 1, d_A, (int)Ne, d_ipiv, d_x, (int)Ne, d_info));
        CUDA_OK(cudaDeviceSynchronize());

        // Read Ipatch and Ignd
        cuDoubleComplex Ih[2];
        CUDA_OK(cudaMemcpy(&Ih[0], d_x + (patchEdge-1), sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
        CUDA_OK(cudaMemcpy(&Ih[1], d_x + (groundEdge-1), sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
        CUDA_OK(cudaFree(d_x));

        std::complex<double> Ipatch(cuCreal(Ih[0]), cuCimag(Ih[0]));
        std::complex<double> Ignd  (cuCreal(Ih[1]), cuCimag(Ih[1]));
        std::complex<double> Iport = Ipatch - Ignd;

        std::complex<double> Zin = std::complex<double>(V0,0.0) / Iport;
        std::complex<double> S11 = (Zin - Z0) / (Zin + Z0);

#if MX_HAS_INTERLEAVED_COMPLEX
        ZinOut[fi].real = Zin.real(); ZinOut[fi].imag = Zin.imag();
        S11Out[fi].real = S11.real(); S11Out[fi].imag = S11.imag();
#else
        ZinRe[fi]=Zin.real(); ZinIm[fi]=Zin.imag();
        S11Re[fi]=S11.real(); S11Im[fi]=S11.imag();
#endif
    }

    // Cleanup
    if(d_work) CUDA_OK(cudaFree(d_work));
    CUSOLVER_OK(cusolverDnDestroy(solver));

    CUDA_OK(cudaFree(d_A));
    CUDA_OK(cudaFree(d_b));
    CUDA_OK(cudaFree(d_ipiv));
    CUDA_OK(cudaFree(d_info));

    CUDA_OK(cudaFree(d_pos1)); CUDA_OK(cudaFree(d_pos2));
    CUDA_OK(cudaFree(d_f1));   CUDA_OK(cudaFree(d_f2));
    CUDA_OK(cudaFree(d_div1)); CUDA_OK(cudaFree(d_div2));
    CUDA_OK(cudaFree(d_w1));   CUDA_OK(cudaFree(d_w2));
    CUDA_OK(cudaFree(d_t1));   CUDA_OK(cudaFree(d_t2));
    CUDA_OK(cudaFree(d_asoft));

    mxFree(triCent); mxFree(triArea); mxFree(asoftTri);
    mxFree(pos1); mxFree(pos2); mxFree(f1); mxFree(f2);
    mxFree(div1); mxFree(div2); mxFree(w1); mxFree(w2);
    mxFree(t1); mxFree(t2);
    mxFree(ecent); mxFree(rhs_r); mxFree(tmp);
    mxFree(b_h);
}
