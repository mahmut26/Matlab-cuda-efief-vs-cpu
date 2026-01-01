#include "mex.h"
#include "matrix.h"

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <math.h>

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

// exp(-j*(kR)) where k = kr + j*ki  ==>  exp(-(j*kr - ki)R) = exp(ki*R) * (cos(krR) - j sin(krR))
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

// Z kernel (centroid 1-pt)
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

            if(tm == tn){
                double a = asoftTri[tm-1];  // tri index 1-based
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

    // column-major
    Z[m + n*Ne] = Zmn;
}

// -------- Helpers (MATLAB old/new) --------
static inline double get_real_scalar(const mxArray* a){
    return mxGetScalar(a);
}
static inline cuDoubleComplex get_complex_scalar(const mxArray* a){
    double re = mxGetScalar(a);
    double im = 0.0;
    if(mxIsComplex(a)){
        double* pi = mxGetPi(a);
        if(pi) im = pi[0];
    }
    return make_cuDoubleComplex(re, im);
}
static void check(bool cond, const char* id, const char* msg){
    if(!cond) mexErrMsgIdAndTxt(id, msg);
}

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    // Inputs (17):
    // 0 V [Nv x 3] double
    // 1 F [Nt x 3] int32
    // 2 E [Ne x 2] int32
    // 3 plusTri [Ne x 1] int32
    // 4 minusTri [Ne x 1] int32
    // 5 plusSign [Ne x 1] int32
    // 6 minusSign [Ne x 1] int32
    // 7 rp [Ne x 3] double
    // 8 rm [Ne x 3] double
    // 9 Ap [Ne x 1] double
    // 10 Am [Ne x 1] double
    // 11 len [Ne x 1] double
    // 12 k (complex scalar)   (omega ile uyumlu şekilde dışarıda hesaplayıp yolla)
    // 13 omega (double)
    // 14 mu (double)
    // 15 eps (complex scalar)
    // 16 exc = [patchEdge, groundEdge, V0, spreadRadius, selfAlpha] double vector length 5
    //
    // Outputs:
    // 0 Z [Ne x Ne] complex double
    // 1 Vrhs [Ne x 1] complex double

    check(nrhs==17, "mex:args", "Expected 17 inputs.");
    check(nlhs==2,  "mex:args", "Expected 2 outputs [Z,Vrhs].");

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

    cuDoubleComplex k_c   = get_complex_scalar(prhs[12]);
    double omega          = get_real_scalar(prhs[13]);
    double mu             = get_real_scalar(prhs[14]);
    cuDoubleComplex eps_c = get_complex_scalar(prhs[15]);

    const mxArray* excMx = prhs[16];

    check(mxIsDouble(Vmx) && mxGetN(Vmx)==3, "mex:V", "V must be [Nv x 3] double.");
    check(mxIsInt32(Fmx)  && mxGetN(Fmx)==3, "mex:F", "F must be [Nt x 3] int32.");
    check(mxIsInt32(Emx)  && mxGetN(Emx)==2, "mex:E", "E must be [Ne x 2] int32.");

    mwSize Nv = mxGetM(Vmx);
    mwSize Nt = mxGetM(Fmx);
    mwSize Ne = mxGetM(Emx);

    check(mxIsInt32(plusTriMx)  && mxGetM(plusTriMx)==Ne,  "mex:plusTri",  "plusTri must be int32 [Ne x 1].");
    check(mxIsInt32(minusTriMx) && mxGetM(minusTriMx)==Ne, "mex:minusTri", "minusTri must be int32 [Ne x 1].");
    check(mxIsInt32(plusSignMx) && mxGetM(plusSignMx)==Ne, "mex:plusSign", "plusSign must be int32 [Ne x 1].");
    check(mxIsInt32(minusSignMx)&& mxGetM(minusSignMx)==Ne,"mex:minusSign","minusSign must be int32 [Ne x 1].");

    check(mxIsDouble(rpMx) && mxGetM(rpMx)==Ne && mxGetN(rpMx)==3, "mex:rp", "rp must be [Ne x 3] double.");
    check(mxIsDouble(rmMx) && mxGetM(rmMx)==Ne && mxGetN(rmMx)==3, "mex:rm", "rm must be [Ne x 3] double.");

    check(mxIsDouble(ApMx) && mxGetNumberOfElements(ApMx)==Ne, "mex:Ap", "Ap must be [Ne x 1] double.");
    check(mxIsDouble(AmMx) && mxGetNumberOfElements(AmMx)==Ne, "mex:Am", "Am must be [Ne x 1] double.");
    check(mxIsDouble(lenMx)&& mxGetNumberOfElements(lenMx)==Ne,"mex:len","len must be [Ne x 1] double.");

    check(mxIsDouble(excMx) && mxGetNumberOfElements(excMx)>=5, "mex:exc", "exc must be double vector len>=5: [patchEdge, groundEdge, V0, spreadRadius, selfAlpha].");

    double* excv = mxGetPr(excMx);
    int patchEdge  = (int)excv[0];
    int groundEdge = (int)excv[1];
    double V0      = excv[2];
    double spread  = excv[3];
    double selfAlpha = excv[4];

    check(patchEdge>=1 && patchEdge<=(int)Ne, "mex:port", "patchEdge out of range.");
    check(groundEdge>=1 && groundEdge<=(int)Ne, "mex:port", "groundEdge out of range.");

    const double* Vh = mxGetPr(Vmx);
    const int* Fh = (const int*)mxGetData(Fmx);
    const int* Eh = (const int*)mxGetData(Emx);

    const int* plusTri  = (const int*)mxGetData(plusTriMx);
    const int* minusTri = (const int*)mxGetData(minusTriMx);
    const int* plusSign = (const int*)mxGetData(plusSignMx);
    const int* minusSign= (const int*)mxGetData(minusSignMx);

    const double* rpH = mxGetPr(rpMx);
    const double* rmH = mxGetPr(rmMx);
    const double* ApH = mxGetPr(ApMx);
    const double* AmH = mxGetPr(AmMx);
    const double* lenH= mxGetPr(lenMx);

    // ---- Precompute triangle centroid & area & asoft (host)
    d3* triCent = (d3*)mxMalloc(Nt*sizeof(d3));
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

        // area
        d3 e1 = sub3(r2,r1);
        d3 e2 = sub3(r3,r1);
        double cx = e1.y*e2.z - e1.z*e2.y;
        double cy = e1.z*e2.x - e1.x*e2.z;
        double cz = e1.x*e2.y - e1.y*e2.x;
        double area2 = sqrt(cx*cx + cy*cy + cz*cz); // 2A
        double A = 0.5 * area2;

        // self softening
        asoftTri[t] = selfAlpha * sqrt(A / PI_D);
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

        pos1[e] = (tp>0) ? triCent[tp-1] : make_d3(0,0,0);
        pos2[e] = (tm>0) ? triCent[tm-1] : make_d3(0,0,0);

        // weights (centroid approx) — burada sen Ap/Am kullandığın için w=A alınmış
        w1[e] = (tp>0) ? ApH[e] : 0.0;
        w2[e] = (tm>0) ? AmH[e] : 0.0;

        double l = lenH[e];

        d3 rp = make_d3(rpH[e + 0*Ne], rpH[e + 1*Ne], rpH[e + 2*Ne]);
        d3 rm = make_d3(rmH[e + 0*Ne], rmH[e + 1*Ne], rmH[e + 2*Ne]);

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

    // ---- Build Vrhs on host (gaussian differential on same layer)
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

    double* Vr = (double*)mxMalloc(Ne*sizeof(double));
    double* tmp = (double*)mxMalloc(Ne*sizeof(double));
    for(mwSize e=0;e<Ne;++e) Vr[e]=0.0;

    auto build_gauss = [&](int edge1based, double amp){
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
        check(sumw>1e-30, "mex:rhs", "spreadRadius too small (no weights).");
        for(mwSize e=0;e<Ne;++e) Vr[e] += amp*(tmp[e]/sumw);
    };

    build_gauss(patchEdge,  +V0/2.0);
    build_gauss(groundEdge, -V0/2.0);

    // ---- alpha / beta
    // alpha = j*omega*mu
    cuDoubleComplex alpha = make_cuDoubleComplex(0.0, omega*mu);

    // beta  = 1/(j*omega*eps)
    // j*omega*eps = ( -omega*Im(eps) ) + j*( omega*Re(eps) )
    double er = cuCreal(eps_c);
    double ei = cuCimag(eps_c);
    cuDoubleComplex denom = make_cuDoubleComplex(-omega*ei, omega*er);
    double denom_abs2 = cuCreal(denom)*cuCreal(denom) + cuCimag(denom)*cuCimag(denom);
    cuDoubleComplex beta  = make_cuDoubleComplex( cuCreal(denom)/denom_abs2, -cuCimag(denom)/denom_abs2 );

    // ---- Device alloc/copy
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

    cuDoubleComplex* d_Z;
    CUDA_OK(cudaMalloc(&d_Z, (size_t)Ne*(size_t)Ne*sizeof(cuDoubleComplex)));

    dim3 block(16,16);
    dim3 grid((unsigned)((Ne + block.x - 1)/block.x),
              (unsigned)((Ne + block.y - 1)/block.y));

    kernel_Z_centroid<<<grid,block>>>(d_Z, (int)Ne,
                                      d_pos1,d_pos2,d_f1,d_f2,
                                      d_div1,d_div2,d_w1,d_w2,
                                      d_t1,d_t2,d_asoft,
                                      k_c, alpha, beta);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaDeviceSynchronize());

    // ---- Copy Z back
    cuDoubleComplex* Zh = (cuDoubleComplex*)mxMalloc((size_t)Ne*(size_t)Ne*sizeof(cuDoubleComplex));
    CUDA_OK(cudaMemcpy(Zh, d_Z, (size_t)Ne*(size_t)Ne*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));

    // ---- Outputs: NON-INTERLEAVED COMPLEX (old MATLAB safe)
    plhs[0] = mxCreateDoubleMatrix((mwSize)Ne, (mwSize)Ne, mxCOMPLEX);
    double* Zre = mxGetPr(plhs[0]);
    double* Zim = mxGetPi(plhs[0]);

    mwSize NN = (mwSize)Ne*(mwSize)Ne;
    for(mwSize i=0;i<NN;++i){
        Zre[i] = cuCreal(Zh[i]);
        Zim[i] = cuCimag(Zh[i]);
    }

    plhs[1] = mxCreateDoubleMatrix((mwSize)Ne, 1, mxCOMPLEX);
    double* Vre = mxGetPr(plhs[1]);
    double* Vim = mxGetPi(plhs[1]);
    for(mwSize e=0;e<Ne;++e){
        Vre[e] = Vr[e];
        Vim[e] = 0.0;
    }

    // ---- Cleanup
    CUDA_OK(cudaFree(d_Z));
    CUDA_OK(cudaFree(d_pos1)); CUDA_OK(cudaFree(d_pos2));
    CUDA_OK(cudaFree(d_f1));   CUDA_OK(cudaFree(d_f2));
    CUDA_OK(cudaFree(d_div1)); CUDA_OK(cudaFree(d_div2));
    CUDA_OK(cudaFree(d_w1));   CUDA_OK(cudaFree(d_w2));
    CUDA_OK(cudaFree(d_t1));   CUDA_OK(cudaFree(d_t2));
    CUDA_OK(cudaFree(d_asoft));

    mxFree(Zh);
    mxFree(triCent); mxFree(asoftTri);
    mxFree(pos1); mxFree(pos2); mxFree(f1); mxFree(f2);
    mxFree(div1); mxFree(div2); mxFree(w1); mxFree(w2);
    mxFree(t1); mxFree(t2);
    mxFree(ecent); mxFree(Vr); mxFree(tmp);
}
