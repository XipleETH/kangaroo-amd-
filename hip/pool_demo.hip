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
    if(role==0){ u64 s=(u64)i*((1ULL<<RANGE_BITS)/HERD)+1ULL; jpt P=scalar_mul_G_jac(fe_set(s,0,0,0)); kx[i]=P.x; ky[i]=P.y; kd[i]=s; }  // +1: index 0 must not be O
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

// Validate a DP endpoint on the GPU: a real tame DP satisfies dist*G == P; a real
// wild DP satisfies Q + dist*G == P. Plus the DP mask. This is the TRUSTLESS layer:
// it deterministically rejects garbage/fabricated DPs (their dist*G != their x).
__global__ void k_verify(const DP* dps, int n, fe Qx, fe Qy, u64 dpmask, unsigned char* ok){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    DP d=dps[i];
    if((d.x[0]&dpmask)!=0ULL){ ok[i]=0; return; }                 // fails the DP mask
    if(d.dist==0ULL){ ok[i]=0; return; }                          // dist 0 -> point at infinity: forgery guard
    jpt P=scalar_mul_G_jac(fe_set(d.dist,0,0,0));
    if(d.type==1){ jpt Q; Q.inf=false; Q.x=Qx; Q.y=Qy; P=j_add(Q,P); }  // wild: + Q
    if(P.inf){ ok[i]=0; return; }                                 // degenerate point, never a real DP
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

// A thief who copies another client's DP file and relabels it as its own, to steal share.
// (self-asserted `client` field — nothing binds a DP to its producer without signatures.)
static int run_replay(int id, const char* src, const char* out){
    setvbuf(stdout,NULL,_IONBF,0);
    std::vector<DP> recs; FILE* f=fopen(src,"rb"); if(!f){ printf("cannot open %s\n",src); return 1; }
    DP r; while(fread(&r,sizeof(DP),1,f)==1){ r.client=(u32)id; recs.push_back(r); } fclose(f);
    FILE* g=fopen(out,"wb"); if(!g){ printf("cannot open %s\n",out); return 1; }
    fwrite(recs.data(),sizeof(DP),recs.size(),g); fclose(g);
    printf("replay: relabeled %zu DPs from %s as client %d -> %s\n", recs.size(), src, id, out);
    return 0;
}
// Dedup key = full (x, type, dist); ties broken by load order so the FIRST submitter wins.
static const DP* G_all;
static int idxcmp(const void* pa, const void* pb){
    int ia=*(const int*)pa, ib=*(const int*)pb; const DP&a=G_all[ia]; const DP&b=G_all[ib];
    for(int j=0;j<4;++j){ if(a.x[j]!=b.x[j]) return a.x[j]<b.x[j]?-1:1; }
    if(a.type!=b.type) return a.type<b.type?-1:1;
    if(a.dist!=b.dist) return a.dist<b.dist?-1:1;
    return ia<ib?-1:(ia>ib?1:0);
}
static bool tuple_eq(const DP&a,const DP&b){ return xeq(a,b)&&a.type==b.type&&a.dist==b.dist; }
static int u32cmp(const void* a, const void* b){ u32 x=*(const u32*)a,y=*(const u32*)b; return x<y?-1:(x>y?1:0); }
static int find_id(const u32* uid, int nc, u32 c){ int lo=0,hi=nc-1;
    while(lo<=hi){ int m=(lo+hi)>>1; if(uid[m]==c)return m; if(uid[m]<c)lo=m+1; else hi=m-1; } return -1; }

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

    // Compact the (arbitrary, self-asserted) client ids so accounting has NO fixed cap and no
    // huge-id allocation: sort-unique the ids, then index counts by position.
    std::vector<u32> ids(n); for(int i=0;i<n;++i) ids[i]=all[i].client;
    qsort(ids.data(), n, sizeof(u32), u32cmp);
    std::vector<u32> uid; for(int i=0;i<n;++i) if(i==0||ids[i]!=ids[i-1]) uid.push_back(ids[i]);
    int NC=(int)uid.size();
    // Per-client accounting: endpoint-VALID, then DEDUPED by (x,type,dist) so a client cannot
    // inflate its share by repeating its own DPs or replaying another's. Each unique DP is
    // credited to its FIRST submitter (load order here stands in for receive order).
    std::vector<long> vc(NC,0), ic(NC,0), uc(NC,0); long totv=0, totu=0;
    for(int i=0;i<n;++i){ int c=find_id(uid.data(),NC,all[i].client); if(ok[i]){vc[c]++; totv++;} else ic[c]++; }
    std::vector<int> vidx; vidx.reserve(totv); for(int i=0;i<n;++i) if(ok[i]) vidx.push_back(i);
    G_all=all.data(); if(!vidx.empty()) qsort(vidx.data(), vidx.size(), sizeof(int), idxcmp);
    for(size_t r=0;r<vidx.size();++r)
        if(r==0 || !tuple_eq(all[vidx[r]], all[vidx[r-1]])){ int c=find_id(uid.data(),NC,all[vidx[r]].client); uc[c]++; totu++; }
    printf("\nContribution (endpoint-valid, then deduped — only UNIQUE work earns a share):\n");
    printf("  client   valid       unique      invalid     share\n");
    for(int c=0;c<NC;++c)
        printf("  %-6u   %-10ld  %-10ld  %-10ld  %5.1f%%\n", uid[c], vc[c], uc[c], ic[c], totu?100.0*uc[c]/totu:0.0);
    printf("  total valid = %ld, unique credited = %ld  (dups & fakes earn nothing)\n", totv, totu);

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
    if(argc==5 && strcmp(argv[1],"replay")==0)
        return run_replay(atoi(argv[2]), argv[3], argv[4]);
    if(argc>=3 && strcmp(argv[1],"detect")==0)
        return run_detect(argc-2, argv+2);
    printf("usage:\n  %s client <id> <tame|wild> <out.dp> <dispatches>\n"
           "  %s faker  <id> <out.dp> <count>\n  %s replay <id> <src.dp> <out.dp>\n"
           "  %s detect <f0.dp> <f1.dp> ...\n", argv[0], argv[0], argv[0], argv[0]);
    return 2;
}
