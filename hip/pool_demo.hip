// pool_demo.hip — proves cross-client DP pooling with ONE AMD card (no Nvidia needed).
//
//   client 0 runs ONLY tame kangaroos -> writes its DPs to a file
//   client 1 runs ONLY wild kangaroos -> writes its DPs to a file
//   NEITHER can solve alone. The detector merges both files, finds a
//   tame(client0) <-> wild(client1) collision, recovers k, and reports WHICH
//   clients contributed (the attribution a fair split needs).
//
// This is the pool's core mechanic. AMD+Nvidia would just be more clients.
//
// Build: hipcc -O3 --offload-arch=gfx1101 pool_demo.hip -o pool_demo.exe
// Run:   pool_demo client 0 tame c0.dp 40
//        pool_demo client 1 wild c1.dp 40
//        pool_demo detect c0.dp c1.dp
#include "secp256k1.h"
#include "ec.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define CHECK(c) do{ hipError_t e=(c); if(e!=hipSuccess){ printf("HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e)); return 1; } }while(0)

// ---- CANONICAL POOL SPEC (every client MUST use these identical values) ----
#define RANGE_BITS 40
#define N_JUMPS    32
#define HERD       262144            // kangaroos per client (all one type)
#define GN         16
#define NUM_THREADS (HERD / GN)
#define DP_BITS    12
#define STEPS      256
#define MAX_DP     (1u << 21)
// Canonical jump scalars: deterministic LCG from a FIXED published seed.
static void canonical_jumps(u64 js[N_JUMPS]) {
    u64 rng = 0x1234567ULL, span = 1ULL << (RANGE_BITS/2 + 1);
    for (int i=0;i<N_JUMPS;++i){ rng = rng*6364136223846793005ULL + 1442695040888963407ULL; js[i]=1ULL+(rng%span); }
}
// Demo target: Q = k*G for a known test key near the middle of the range.
#define KTEST_LO ((1ULL << (RANGE_BITS-1)) + 0x12345ULL)

typedef unsigned int u32;
struct DP { u64 x[4]; u64 dist; u32 type; u32 client; };  // 48 bytes, wire/file record

__global__ void k_mkQ(fe* Qx, fe* Qy, u64 k0){ jpt Q=scalar_mul_G_jac(fe_set(k0,0,0,0)); *Qx=Q.x; *Qy=Q.y; }
__global__ void k_mkjumps(fe* jx, fe* jy, const u64* js){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=N_JUMPS)return;
    jpt J=scalar_mul_G_jac(fe_set(js[i],0,0,0)); jx[i]=J.x; jy[i]=J.y; }
// role 0 = tame (spread over range), role 1 = wild (near Q)
__global__ void k_mkkang(fe* kx, fe* ky, u64* kd, const fe* Qx, const fe* Qy, int role){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=HERD)return;
    if(role==0){ u64 s=(u64)i*((1ULL<<RANGE_BITS)/HERD); jpt P=scalar_mul_G_jac(fe_set(s,0,0,0)); kx[i]=P.x; ky[i]=P.y; kd[i]=s; }
    else { u64 w=(u64)i; jpt off=scalar_mul_G_jac(fe_set(w,0,0,0)); jpt Q; Q.inf=false; Q.x=*Qx; Q.y=*Qy;
           jpt P=j_add(Q,off); kx[i]=P.x; ky[i]=P.y; kd[i]=w; }
}
// global-backed batch-inversion walk (same as kangaroo.hip), tags every DP with `type`
__global__ void k_walk(fe* kx, fe* ky, u64* kd, const fe* jx, const fe* jy, const u64* js,
                       fe* PRE, fe* dpx, u64* dpd, unsigned* dpcnt, int steps, u64 dpmask, u32 type, unsigned char* dpt){
    int t=blockIdx.x*blockDim.x+threadIdx.x; if(t>=NUM_THREADS)return;
    for(int s=0;s<steps;++s){
        fe prod=fe_one();
        for(int j=0;j<GN;++j){ int kid=t+j*NUM_THREADS; fe x=kx[kid]; int idx=(int)(x.n[0]%(u64)N_JUMPS);
            prod=fe_mul(prod, fe_sub(jx[idx],x)); PRE[j*NUM_THREADS+t]=prod; }
        fe inv=fe_inv(prod);
        for(int j=GN-1;j>=0;--j){ int kid=t+j*NUM_THREADS; fe x=kx[kid], y=ky[kid]; int idx=(int)(x.n[0]%(u64)N_JUMPS);
            fe dxinv=(j>0)?fe_mul(inv,PRE[(j-1)*NUM_THREADS+t]):inv; inv=fe_mul(inv,fe_sub(jx[idx],x));
            fe lam=fe_mul(fe_sub(jy[idx],y),dxinv); fe nx=fe_sub(fe_sub(fe_sqr(lam),x),jx[idx]);
            fe ny=fe_sub(fe_mul(lam,fe_sub(x,nx)),y); kx[kid]=nx; ky[kid]=ny; u64 d=kd[kid]+js[idx]; kd[kid]=d;
            if((nx.n[0]&dpmask)==0ULL){ unsigned p=atomicAdd(dpcnt,1u); if(p<MAX_DP){ dpx[p]=nx; dpd[p]=d; dpt[p]=(unsigned char)type; } } }
    }
}

#define MAXC 64
// Validate a DP endpoint on the GPU: a real tame DP satisfies dist*G == P; a real
// wild DP satisfies Q + dist*G == P. Plus the DP mask. This is the TRUSTLESS layer:
// it deterministically rejects garbage/fabricated DPs (their dist*G != their x).
__global__ void k_verify(const DP* dps, int n, fe Qx, fe Qy, u64 dpmask, unsigned char* ok){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    DP d=dps[i];
    if((d.x[0]&dpmask)!=0ULL){ ok[i]=0; return; }                 // fails the DP mask
    jpt P=scalar_mul_G_jac(fe_set(d.dist,0,0,0));
    if(d.type==1){ jpt Q; Q.inf=false; Q.x=Qx; Q.y=Qy; P=j_add(Q,P); }  // wild: + Q
    fe X=fe_set(d.x[0],d.x[1],d.x[2],d.x[3]);
    ok[i]=fe_eq(P.x,X)?1:0;
}

static int run_client(int id, int role, const char* outfile, int dispatches){
    setvbuf(stdout,NULL,_IONBF,0);
    printf("client %d (%s): setup...\n", id, role==0?"tame":"wild");
    fe *Qx,*Qy,*jx,*jy,*kx,*ky,*dpx,*PRE; u64 *js,*kd,*dpd; unsigned char *dpt; unsigned *dpcnt;
    u64 h_js[N_JUMPS]; canonical_jumps(h_js);
    CHECK(hipMalloc(&Qx,sizeof(fe))); CHECK(hipMalloc(&Qy,sizeof(fe)));
    CHECK(hipMalloc(&jx,N_JUMPS*sizeof(fe))); CHECK(hipMalloc(&jy,N_JUMPS*sizeof(fe))); CHECK(hipMalloc(&js,N_JUMPS*sizeof(u64)));
    CHECK(hipMalloc(&kx,HERD*sizeof(fe))); CHECK(hipMalloc(&ky,HERD*sizeof(fe))); CHECK(hipMalloc(&kd,HERD*sizeof(u64)));
    CHECK(hipMalloc(&PRE,(size_t)HERD*sizeof(fe)));
    CHECK(hipMalloc(&dpx,MAX_DP*sizeof(fe))); CHECK(hipMalloc(&dpd,MAX_DP*sizeof(u64)));
    CHECK(hipMalloc(&dpt,MAX_DP)); CHECK(hipMalloc(&dpcnt,4));
    CHECK(hipMemcpy(js,h_js,N_JUMPS*sizeof(u64),hipMemcpyHostToDevice));
    hipLaunchKernelGGL(k_mkQ,dim3(1),dim3(1),0,0,Qx,Qy,(u64)KTEST_LO);
    hipLaunchKernelGGL(k_mkjumps,dim3(1),dim3(N_JUMPS),0,0,jx,jy,js);
    hipLaunchKernelGGL(k_mkkang,dim3((HERD+255)/256),dim3(256),0,0,kx,ky,kd,Qx,Qy,role);
    CHECK(hipDeviceSynchronize());

    std::vector<DP> recs; const u64 dpmask=(1ULL<<DP_BITS)-1ULL;
    fe* hx=new fe[MAX_DP]; u64* hd=new u64[MAX_DP]; unsigned char* ht=new unsigned char[MAX_DP];
    for(int it=0; it<dispatches; ++it){
        CHECK(hipMemset(dpcnt,0,4));
        hipLaunchKernelGGL(k_walk,dim3((NUM_THREADS+255)/256),dim3(256),0,0,kx,ky,kd,jx,jy,js,PRE,dpx,dpd,dpcnt,STEPS,dpmask,(u32)role,dpt);
        CHECK(hipDeviceSynchronize());
        unsigned cnt=0; CHECK(hipMemcpy(&cnt,dpcnt,4,hipMemcpyDeviceToHost)); if(cnt>MAX_DP)cnt=MAX_DP;
        if(cnt){ CHECK(hipMemcpy(hx,dpx,cnt*sizeof(fe),hipMemcpyDeviceToHost));
                 CHECK(hipMemcpy(hd,dpd,cnt*sizeof(u64),hipMemcpyDeviceToHost));
                 CHECK(hipMemcpy(ht,dpt,cnt,hipMemcpyDeviceToHost));
                 for(unsigned j=0;j<cnt;++j){ DP r; r.x[0]=hx[j].n[0]; r.x[1]=hx[j].n[1]; r.x[2]=hx[j].n[2]; r.x[3]=hx[j].n[3];
                     r.dist=hd[j]; r.type=ht[j]; r.client=(u32)id; recs.push_back(r); } }
    }
    FILE* f=fopen(outfile,"wb"); if(!f){ printf("cannot open %s\n",outfile); return 1; }
    fwrite(recs.data(),sizeof(DP),recs.size(),f); fclose(f);
    printf("client %d wrote %zu DPs to %s\n", id, recs.size(), outfile);
    return 0;
}

static bool xeq(const DP&a, const DP&b){ return a.x[0]==b.x[0]&&a.x[1]==b.x[1]&&a.x[2]==b.x[2]&&a.x[3]==b.x[3]; }
static int dpcmp(const void* pa, const void* pb){ const DP*a=(const DP*)pa,*b=(const DP*)pb;
    for(int i=0;i<4;++i){ if(a->x[i]!=b->x[i]) return a->x[i]<b->x[i]?-1:1; } return 0; }

// A cheater who submits mask-valid but bogus DPs to inflate its reward share.
// (random x with the low DP_BITS zeroed -> passes the mask; random dist -> endpoint
//  check dist*G==x FAILS, so the validator rejects every one.)
static int run_faker(int id, const char* outfile, int count){
    setvbuf(stdout,NULL,_IONBF,0);
    std::vector<DP> recs; u64 rng=0xF00DULL^(u64)id; const u64 dpmask=(1ULL<<DP_BITS)-1ULL;
    for(int i=0;i<count;++i){ DP r;
        for(int j=0;j<4;++j){ rng=rng*6364136223846793005ULL+1442695040888963407ULL; r.x[j]=rng; }
        r.x[0]&=~dpmask;                                                   // look legit ...
        rng=rng*6364136223846793005ULL+1442695040888963407ULL; r.dist=rng>>20;  // ... bogus dist
        r.type=(u32)(i&1); r.client=(u32)id; recs.push_back(r); }
    FILE* f=fopen(outfile,"wb"); if(!f){ printf("cannot open %s\n",outfile); return 1; }
    fwrite(recs.data(),sizeof(DP),recs.size(),f); fclose(f);
    printf("faker %d wrote %d FAKE DPs (mask-valid, endpoint-bogus) to %s\n", id, count, outfile);
    return 0;
}

static int run_detect(int nfiles, char** files){
    setvbuf(stdout,NULL,_IONBF,0);
    std::vector<DP> all;
    for(int fi=0; fi<nfiles; ++fi){
        FILE* f=fopen(files[fi],"rb"); if(!f){ printf("cannot open %s\n",files[fi]); return 1; }
        DP r; while(fread(&r,sizeof(DP),1,f)==1) all.push_back(r);
        fclose(f);
    }
    int n=(int)all.size(); printf("merged %d DPs from %d files\n", n, nfiles);
    if(n==0){ printf("no DPs\n"); return 1; }

    // The public target Q = k*G (known to everyone; the detector never needs k).
    fe *dQx,*dQy; CHECK(hipMalloc(&dQx,sizeof(fe))); CHECK(hipMalloc(&dQy,sizeof(fe)));
    hipLaunchKernelGGL(k_mkQ,dim3(1),dim3(1),0,0,dQx,dQy,(u64)KTEST_LO); CHECK(hipDeviceSynchronize());
    fe hQx,hQy; CHECK(hipMemcpy(&hQx,dQx,sizeof(fe),hipMemcpyDeviceToHost)); CHECK(hipMemcpy(&hQy,dQy,sizeof(fe),hipMemcpyDeviceToHost));

    // Validate EVERY DP on the GPU (chunked to stay under the 2s Windows TDR).
    DP* dAll; CHECK(hipMalloc(&dAll,(size_t)n*sizeof(DP)));
    CHECK(hipMemcpy(dAll,all.data(),(size_t)n*sizeof(DP),hipMemcpyHostToDevice));
    unsigned char* dok; CHECK(hipMalloc(&dok,n));
    const u64 dpmask=(1ULL<<DP_BITS)-1ULL; const int CH=131072;
    for(int off=0; off<n; off+=CH){ int m=(n-off<CH)?(n-off):CH;
        hipLaunchKernelGGL(k_verify,dim3((m+255)/256),dim3(256),0,0,dAll+off,m,hQx,hQy,dpmask,dok+off);
        CHECK(hipDeviceSynchronize()); }
    std::vector<unsigned char> ok(n); CHECK(hipMemcpy(ok.data(),dok,n,hipMemcpyDeviceToHost));

    // Per-client contribution accounting — only endpoint-VALID DPs count.
    long vc[MAXC]={0}, ic[MAXC]={0}; long totv=0; bool seen[MAXC]={false};
    for(int i=0;i<n;++i){ u32 c=all[i].client; if(c<MAXC){ seen[c]=true; if(ok[i]){vc[c]++; totv++;} else ic[c]++; } }
    printf("\nContribution (only endpoint-verified DPs earn a share; fakes rejected):\n");
    printf("  client   valid       invalid     share\n");
    for(int c=0;c<MAXC;++c) if(seen[c])
        printf("  %-6d   %-10ld  %-10ld  %5.1f%%\n", c, vc[c], ic[c], totv?100.0*vc[c]/totv:0.0);
    printf("  total valid = %ld\n", totv);

    // Collision detection among VALID DPs only.
    std::vector<DP> valid; valid.reserve(totv);
    for(int i=0;i<n;++i) if(ok[i]) valid.push_back(all[i]);
    if(!valid.empty()) qsort(valid.data(), valid.size(), sizeof(DP), dpcmp);
    for(size_t i=1;i<valid.size();++i){
        if(xeq(valid[i-1],valid[i]) && valid[i-1].type!=valid[i].type){
            DP a=valid[i-1], b=valid[i];
            u64 dt=(a.type==0)?a.dist:b.dist, dw=(a.type==1)?a.dist:b.dist; u64 k=dt-dw;
            u32 ct=(a.type==0?a.client:b.client), cw=(a.type==1?a.client:b.client);
            printf("\n*** CROSS-CLIENT COLLISION (validated DPs) ***\n");
            printf("  tame DP from client %u  +  wild DP from client %u\n", ct, cw);
            printf("  recovered k = 0x%llx  (expected 0x%llx)  -> %s\n",
                   (unsigned long long)k,(unsigned long long)KTEST_LO,k==KTEST_LO?"MATCH":"mismatch");
            printf("\nNote: endpoint validation kills GARBAGE fakes, but a 'smart' faker\n"
                   "(random d, compute d*G, keep if masked) passes it yet isn't on the shared\n"
                   "walk -> that residual is closed by rate-limits/stake, not cryptography.\n");
            return k==KTEST_LO?0:1;
        }
    }
    printf("\nno cross-type collision among valid DPs (run the clients longer)\n");
    return 1;
}

int main(int argc, char** argv){
    if(argc>=2 && strcmp(argv[1],"client")==0 && argc==6)
        return run_client(atoi(argv[2]), strcmp(argv[3],"wild")==0?1:0, argv[4], atoi(argv[5]));
    if(argc==5 && strcmp(argv[1],"faker")==0)
        return run_faker(atoi(argv[2]), argv[3], atoi(argv[4]));
    if(argc>=3 && strcmp(argv[1],"detect")==0)
        return run_detect(argc-2, argv+2);
    printf("usage:\n  %s client <id> <tame|wild> <out.dp> <dispatches>\n"
           "  %s faker  <id> <out.dp> <count>\n  %s detect <f0.dp> <f1.dp> ...\n", argv[0], argv[0], argv[0]);
    return 2;
}
