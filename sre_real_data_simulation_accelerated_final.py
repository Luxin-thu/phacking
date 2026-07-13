from __future__ import annotations

import os
import time
import math
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import norm
import rdata
from numba import njit, prange, set_num_threads

SQRT2 = math.sqrt(2.0)


def load_data(path):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        d = rdata.read_rda(path)
    Y0=np.asarray(d['Y0'],dtype=np.float64)
    Y1=np.asarray(d['Y1'],dtype=np.float64)
    X=np.asarray(d['X'],dtype=np.float64)
    B=np.asarray(d['B'],dtype=np.int64)-1
    K=int(np.asarray(d['K']).ravel()[0]); H=int(np.asarray(d['H']).ravel()[0])
    nh=np.asarray(d['nh_vec'],dtype=np.int64)
    nh1=np.asarray(d['nh1_vec'],dtype=np.int64)
    pi=np.asarray(d['pih_vec'],dtype=np.float64)
    starts=np.zeros(H,dtype=np.int64); starts[1:]=np.cumsum(nh)[:-1]
    return Y0,Y1,X,B,K,H,nh,nh1,pi,starts

def random_assignment_batch(rng, n, n1, batch):
    # Random-key method: exact uniform fixed-size samples.
    keys=rng.random((batch,n),dtype=np.float64)
    idx=np.argpartition(keys,n1-1,axis=1)[:,:n1]
    Z=np.zeros((batch,n),dtype=np.float64)
    Z[np.arange(batch)[:,None],idx]=1.0
    return Z

def calibrate_thresholds(X, nh, nh1, pi, starts, p_re=0.01, draws=100000, seed=12345, batch_size=5000):
    H=len(nh); K=X.shape[1]
    rng=np.random.default_rng(seed)
    a_rem=np.empty(H); t_rep_ss=np.empty(H)
    for h in range(H):
        s=starts[h]; n=nh[h]; n1=nh1[h]; n0=n-n1
        Xh=X[s:s+n]
        total=Xh.sum(0); total2=(Xh*Xh).sum(0)
        invcov=np.linalg.pinv(np.cov(Xh,rowvar=False,ddof=1), rcond=1e-12)
        vals_m=[]; vals_t=[]
        left=draws
        while left:
            b=min(batch_size,left); left-=b
            Z=random_assignment_batch(rng,n,n1,b)
            sx=Z@Xh; sx2=Z@(Xh*Xh)
            mean1=sx/n1; mean0=(total-sx)/n0; delta=mean1-mean0
            m=n*(n1/n)*(n0/n)*np.einsum('bi,ij,bj->b',delta,invcov,delta)
            # R var * (n_g-1)/n_g^2 = centered sumsq / n_g^2
            ss1=np.maximum(sx2-sx*sx/n1,0.0)
            sx0=total-sx; sx20=total2-sx2
            ss0=np.maximum(sx20-sx0*sx0/n0,0.0)
            den=np.sqrt(ss1/(n1*n1)+ss0/(n0*n0))
            tt=np.max(np.abs(delta)/den,axis=1)
            vals_m.append(m); vals_t.append(tt)
        vm=np.concatenate(vals_m); vt=np.concatenate(vals_t)
        # Match R quantile default approximately (linear interpolation).
        a_rem[h]=np.quantile(vm,p_re,method='linear')
        t_rep_ss[h]=np.quantile(vt,p_re,method='linear')
    # Marginal FE-ReP threshold
    rh1=nh1/nh; rh0=1-rh1
    omega=pi*rh1*rh0; omega=omega/omega.sum()
    ch=1/(rh1*rh0)-3
    vals=[]; left=draws
    while left:
        b=min(max(1000,batch_size//2),left); left-=b
        dh=np.empty((b,H,K)); vh=np.empty((b,H,K))
        for h in range(H):
            s=starts[h]; n=nh[h]; n1=nh1[h]; n0=n-n1; Xh=X[s:s+n]
            total=Xh.sum(0); total2=(Xh*Xh).sum(0)
            Z=random_assignment_batch(rng,n,n1,b)
            sx=Z@Xh; sx2=Z@(Xh*Xh)
            sx0=total-sx; sx20=total2-sx2
            d=sx/n1-sx0/n0
            ss1=np.maximum(sx2-sx*sx/n1,0.0)
            ss0=np.maximum(sx20-sx0*sx0/n0,0.0)
            v=ss1/(n1*n1)+ss0/(n0*n0)
            dh[:,h,:]=d; vh[:,h,:]=v
        tau=np.einsum('h,bhk->bk',omega,dh)
        sd2=np.zeros((b,K))
        for h in range(H):
            sd2 += omega[h]**2*(vh[:,h,:]+ch[h]*(dh[:,h,:]-tau)**2/nh[h])
        tmax=np.max(np.abs(tau)/np.sqrt(sd2),axis=1)
        vals.append(tmax)
    t_rep_mn=np.quantile(np.concatenate(vals),p_re,method='linear')
    return a_rem,t_rep_ss,float(t_rep_mn)

def build_subsets(K):
    nmask=1<<K
    sizes=np.zeros(nmask,dtype=np.int64)
    inds=np.full((nmask,K),-1,dtype=np.int64)
    for m in range(nmask):
        arr=[j for j in range(K) if (m>>j)&1]
        sizes[m]=len(arr)
        if arr: inds[m,:len(arr)]=arr
    return sizes,inds

def build_ss_totals(X,Y,nh,starts):
    H=len(nh); K=X.shape[1]
    tx=np.zeros((H,K)); txx=np.zeros((H,K,K)); ty=np.zeros(H); txy=np.zeros((H,K))
    for h in range(H):
        s=starts[h]; e=s+nh[h]; x=X[s:e]; y=Y[s:e]
        tx[h]=x.sum(0); txx[h]=x.T@x; ty[h]=y.sum(); txy[h]=x.T@y
    return tx,txx,ty,txy

def build_mn_controls(X,Y,B,H,sizes,inds):
    N,K=X.shape; nmask=len(sizes); maxp=H+K # intercept + H-1 strata + K
    Cpad=np.zeros((nmask,N,maxp),dtype=np.float64)
    invpad=np.zeros((nmask,maxp,maxp),dtype=np.float64)
    yres=np.zeros((nmask,N),dtype=np.float64)
    ranks=np.zeros(nmask,dtype=np.int64)
    # centered stratum indicators same column space as FE
    pi_counts=np.bincount(B,minlength=H)/N
    S=np.column_stack([(B==h).astype(float)-pi_counts[h] for h in range(1,H)])
    for m in range(nmask):
        q=sizes[m]
        cols=[np.ones(N), *[S[:,j] for j in range(H-1)]]
        for a in range(q): cols.append(X[:,inds[m,a]])
        C=np.column_stack(cols)
        G=C.T@C
        inv=np.linalg.pinv(G,rcond=1e-12)
        yr=Y-C@(inv@(C.T@Y))
        p=C.shape[1]
        Cpad[m,:,:p]=C; invpad[m,:p,:p]=inv; yres[m]=yr; ranks[m]=p
    return Cpad,invpad,yres,ranks

@njit(inline='always')
def rng_next(state):
    x=state
    x ^= (x >> np.uint64(12))
    x ^= (x << np.uint64(25))
    x ^= (x >> np.uint64(27))
    state=x
    return state, x*np.uint64(2685821657736338717)

@njit(inline='always')
def rand_bounded(state,bound):
    state,r=rng_next(state)
    return state,int(r % np.uint64(bound))

@njit
def draw_segment(z,offset,n,n1,index,state):
    for i in range(n):
        index[i]=i; z[offset+i]=0
    for i in range(n1):
        state,jj=rand_bounded(state,n-i)
        j=i+jj
        tmp=index[i]; index[i]=index[j]; index[j]=tmp
        z[offset+index[i]]=1
    return state

@njit
def hacked_p_mn(X,Y,z,sizes,inds,Cpad,invpad,yres,ranks,nh1):
    nmask=sizes.shape[0]; N=X.shape[0]; H=nh1.shape[0]
    n1tot=0
    for h in range(H): n1tot+=nh1[h]
    max_t=0.0
    for m in range(nmask):
        p=ranks[m]
        ctz=np.zeros(p)
        # generic but compiled; p <= 13
        for i in range(N):
            if z[i]==1:
                for a in range(p): ctz[a]+=Cpad[m,i,a]
        v=np.zeros(p)
        for a in range(p):
            for b in range(p): v[a]+=invpad[m,a,b]*ctz[b]
        denom=n1tot
        for a in range(p): denom-=ctz[a]*v[a]
        num=0.0
        for i in range(N):
            if z[i]==1: num+=yres[m,i]
        if denom<=1e-14: continue
        beta=num/denom; var=0.0
        for i in range(N):
            fitz=0.0
            for a in range(p): fitz+=Cpad[m,i,a]*v[a]
            zr=z[i]-fitz
            e=yres[m,i]-beta*zr
            var+=e*e*zr*zr
        var=var/(denom*denom)
        if var>0:
            t=abs(beta)/math.sqrt(var)
            if t>max_t: max_t=t
    return math.erfc(max_t/SQRT2)

@njit
def rem_ok_segment(X,z,offset,n,n1,invcov,threshold):
    K=X.shape[1]; sx=np.zeros(K)
    for ii in range(n):
        i=offset+ii
        if z[i]==1:
            for a in range(K): sx[a]+=X[i,a]
    n0=n-n1; val=0.0
    for a in range(K):
        da=sx[a]/n1 - (-sx[a])/n0 # X centered within stratum
        tmp=0.0
        for c in range(K):
            dc=sx[c]/n1 - (-sx[c])/n0
            tmp+=invcov[a,c]*dc
        val+=da*tmp
    val*=n*(n1/n)*(n0/n)
    return val<=threshold

@njit
def rep_ok_segment(X,z,offset,n,n1,threshold_t):
    K=X.shape[1]; n0=n-n1; sx=np.zeros(K); sx2=np.zeros(K); total=np.zeros(K); total2=np.zeros(K)
    for ii in range(n):
        i=offset+ii
        for a in range(K):
            x=X[i,a]; total[a]+=x; total2[a]+=x*x
            if z[i]==1: sx[a]+=x; sx2[a]+=x*x
    for a in range(K):
        sx0=total[a]-sx[a]; sx20=total2[a]-sx2[a]
        d=sx[a]/n1-sx0/n0
        ss1=sx2[a]-sx[a]*sx[a]/n1; ss0=sx20-sx0*sx0/n0
        den=math.sqrt(max(ss1,0.0)/(n1*n1)+max(ss0,0.0)/(n0*n0))
        if den>0 and abs(d)/den>threshold_t: return False
    return True

@njit
def t_value_mn_max(X,z,nh,nh1,pi,starts):
    H=nh.shape[0]; K=X.shape[1]
    rh1=nh1/nh; rh0=1.0-rh1
    omega=pi*rh1*rh0; omega=omega/np.sum(omega)
    d=np.zeros((H,K)); vh=np.zeros((H,K))
    for h in range(H):
        n=nh[h]; n1=nh1[h]; n0=n-n1; s0=starts[h]
        sx=np.zeros(K); sx2=np.zeros(K); total=np.zeros(K); total2=np.zeros(K)
        for ii in range(n):
            i=s0+ii
            for a in range(K):
                x=X[i,a]; total[a]+=x; total2[a]+=x*x
                if z[i]==1: sx[a]+=x; sx2[a]+=x*x
        for a in range(K):
            sx0=total[a]-sx[a]; sx20=total2[a]-sx2[a]
            d[h,a]=sx[a]/n1-sx0/n0
            ss1=sx2[a]-sx[a]*sx[a]/n1; ss0=sx20-sx0*sx0/n0
            vh[h,a]=max(ss1,0.0)/(n1*n1)+max(ss0,0.0)/(n0*n0)
    max_t=0.0
    for a in range(K):
        tau=0.0
        for h in range(H): tau+=omega[h]*d[h,a]
        sd2=0.0
        for h in range(H):
            ch=1.0/(rh1[h]*rh0[h])-3.0
            diff=d[h,a]-tau
            sd2+=omega[h]*omega[h]*(vh[h,a]+ch*diff*diff/nh[h])
        if sd2>0:
            t=abs(tau)/math.sqrt(sd2)
            if t>max_t: max_t=t
    return max_t

@njit(parallel=True, cache=True)
def analyze_mn_batch(Zs,X,Y,sizes,inds,Cpad,invpad,yres,ranks,nh1):
    out=np.empty(Zs.shape[0])
    for j in prange(Zs.shape[0]):
        out[j]=hacked_p_mn(X,Y,Zs[j],sizes,inds,Cpad,invpad,yres,ranks,nh1)
    return out

@njit
def active_columns_ordered(G, p, tol, active):
    L=np.zeros((p,p))
    r=0
    maxdiag=1.0
    for j in range(p):
        if G[j,j]>maxdiag: maxdiag=G[j,j]
    thresh=tol*maxdiag
    for j in range(p):
        res=G[j,j]
        for a in range(r):
            k=active[a]
            val=G[j,k]
            for b in range(a):
                val-=L[j,b]*L[k,b]
            val/=L[k,a]
            L[j,a]=val
            res-=val*val
        if res>thresh:
            active[r]=j
            L[j,r]=math.sqrt(res)
            r+=1
    return r

@njit
def hacked_p_ss_joint(X,Y,z,nh,nh1,pi,starts,sizes,inds,tx,txx,ty,txy):
    H=nh.shape[0]; K=X.shape[1]; nmask=sizes.shape[0]
    sx1=np.zeros((H,K)); sxx1=np.zeros((H,K,K)); sy1=np.zeros(H); sxy1=np.zeros((H,K))
    for h in range(H):
        s0=starts[h]; n=nh[h]
        for ii in range(n):
            i=s0+ii
            if z[i]==1:
                yi=Y[i]; sy1[h]+=yi
                for a in range(K):
                    xa=X[i,a]; sx1[h,a]+=xa; sxy1[h,a]+=xa*yi
                    for c in range(K): sxx1[h,a,c]+=xa*X[i,c]
    max_t=0.0
    maxp=2+2*K
    for m in range(nmask):
        q=sizes[m]; p=2+2*q; ate=0.0; vv=0.0
        # map local design columns to full source: 0=1,1=z, 2..=X, 2+K..=ZX
        cmap=np.empty(p,dtype=np.int64); cmap[0]=0; cmap[1]=1
        for a in range(q):
            j=inds[m,a]; cmap[2+a]=2+j; cmap[2+q+a]=2+K+j
        for h in range(H):
            Gfull=np.zeros((maxp,maxp)); gyfull=np.zeros(maxp)
            n=nh[h]; n1=nh1[h]
            Gfull[0,0]=n; Gfull[0,1]=n1; Gfull[1,0]=n1; Gfull[1,1]=n1
            gyfull[0]=ty[h]; gyfull[1]=sy1[h]
            for a in range(K):
                xa=tx[h,a]; sxa=sx1[h,a]
                Gfull[0,2+a]=xa; Gfull[2+a,0]=xa
                Gfull[0,2+K+a]=sxa; Gfull[2+K+a,0]=sxa
                Gfull[1,2+a]=sxa; Gfull[2+a,1]=sxa
                Gfull[1,2+K+a]=sxa; Gfull[2+K+a,1]=sxa
                gyfull[2+a]=txy[h,a]; gyfull[2+K+a]=sxy1[h,a]
                for c in range(K):
                    xx=txx[h,a,c]; sxx=sxx1[h,a,c]
                    Gfull[2+a,2+c]=xx
                    Gfull[2+a,2+K+c]=sxx
                    Gfull[2+K+a,2+c]=sxx
                    Gfull[2+K+a,2+K+c]=sxx
            G=np.empty((p,p)); gy=np.empty(p)
            for a in range(p):
                gy[a]=gyfull[cmap[a]]
                for c in range(p): G[a,c]=Gfull[cmap[a],cmap[c]]
            active=np.empty(p,dtype=np.int64)
            r=active_columns_ordered(G,p,1e-10,active)
            # target Z is local col 1 and should always be active
            pos=-1
            A=np.empty((r,r)); rhs=np.zeros((r,2))
            for a in range(r):
                ia=active[a]
                if ia==1: pos=a
                rhs[a,0]=gy[ia]
                rhs[a,1]=1.0 if ia==1 else 0.0
                for c in range(r): A[a,c]=G[ia,active[c]]
            if pos<0: continue
            sol=np.linalg.solve(A,rhs)
            beta=sol[:,0]; v=sol[:,1]
            tau=beta[pos]; se2=0.0; s0=starts[h]
            for ii in range(n):
                i=s0+ii; pred=0.0; w=0.0
                for a in range(r):
                    local=active[a]; src=cmap[local]
                    if src==0: val=1.0
                    elif src==1: val=float(z[i])
                    elif src<2+K: val=X[i,src-2]
                    else: val=X[i,src-(2+K)]*z[i]
                    pred+=val*beta[a]; w+=val*v[a]
                resid=Y[i]-pred; se2+=resid*resid*w*w
            ate+=pi[h]*tau; vv+=pi[h]*pi[h]*se2
        if vv>0.0:
            tt=abs(ate)/math.sqrt(vv)
            if tt>max_t: max_t=tt
    return math.erfc(max_t/SQRT2)

@njit(parallel=True, cache=True)
def analyze_ss_joint_batch(Zs,X,Y,nh,nh1,pi,starts,sizes,inds,tx,txx,ty,txy):
    out=np.empty(Zs.shape[0])
    for j in prange(Zs.shape[0]):
        out[j]=hacked_p_ss_joint(X,Y,Zs[j],nh,nh1,pi,starts,sizes,inds,tx,txx,ty,txy)
    return out

@njit(parallel=True, cache=True)
def generate_sre_numba(seeds, nh, nh1, starts, N):
    Bn=seeds.shape[0]; H=nh.shape[0]; maxn=np.max(nh)
    Zs=np.zeros((Bn,N),dtype=np.uint8)
    for j in prange(Bn):
        state=np.uint64(seeds[j] | np.uint64(1)); idx=np.empty(maxn,dtype=np.int64)
        for h in range(H):
            state=draw_segment(Zs[j],starts[h],nh[h],nh1[h],idx,state)
    return Zs

@njit(parallel=True, cache=True)
def generate_rem_ss_numba(seeds, X, nh, nh1, starts, invcovs, thresholds):
    Bn=seeds.shape[0]; H=nh.shape[0]; N=X.shape[0]; maxn=np.max(nh)
    Zs=np.zeros((Bn,N),dtype=np.uint8); tries=np.zeros(Bn,dtype=np.int64)
    for j in prange(Bn):
        state=np.uint64(seeds[j] | np.uint64(1)); idx=np.empty(maxn,dtype=np.int64)
        for h in range(H):
            ok=False
            while not ok:
                state=draw_segment(Zs[j],starts[h],nh[h],nh1[h],idx,state)
                tries[j]+=1
                ok=rem_ok_segment(X,Zs[j],starts[h],nh[h],nh1[h],invcovs[h],thresholds[h])
    return Zs,tries

@njit(parallel=True, cache=True)
def generate_rep_ss_numba(seeds, X, nh, nh1, starts, thresholds_t):
    Bn=seeds.shape[0]; H=nh.shape[0]; N=X.shape[0]; maxn=np.max(nh)
    Zs=np.zeros((Bn,N),dtype=np.uint8); tries=np.zeros(Bn,dtype=np.int64)
    for j in prange(Bn):
        state=np.uint64(seeds[j] | np.uint64(1)); idx=np.empty(maxn,dtype=np.int64)
        for h in range(H):
            ok=False
            while not ok:
                state=draw_segment(Zs[j],starts[h],nh[h],nh1[h],idx,state)
                tries[j]+=1
                ok=rep_ok_segment(X,Zs[j],starts[h],nh[h],nh1[h],thresholds_t[h])
    return Zs,tries

@njit(parallel=True, cache=True)
def generate_rep_mn_numba(seeds, X, nh, nh1, pi, starts, threshold_t):
    Bn=seeds.shape[0]; H=nh.shape[0]; N=X.shape[0]; maxn=np.max(nh)
    Zs=np.zeros((Bn,N),dtype=np.uint8); tries=np.zeros(Bn,dtype=np.int64)
    for j in prange(Bn):
        state=np.uint64(seeds[j] | np.uint64(1)); idx=np.empty(maxn,dtype=np.int64)
        ok=False
        while not ok:
            for h in range(H):
                state=draw_segment(Zs[j],starts[h],nh[h],nh1[h],idx,state)
            tries[j]+=1
            ok=t_value_mn_max(X,Zs[j],nh,nh1,pi,starts)<=threshold_t
    return Zs,tries

# -----------------------------------------------------------------------------
# Final accelerated driver
# -----------------------------------------------------------------------------
def main_accelerated():
    import json

    workdir = Path(os.environ.get("WORKDIR", "/mnt/data"))
    data_path = Path(
        os.environ.get(
            "DATA_PATH",
            str(workdir / "SRE_real_data_sharp_null.Rdata"),
        )
    )
    num_rep = int(os.environ.get("NUM_REP", "100000"))
    threshold_draws = int(os.environ.get("NUM_THRESHOLD_DRAWS", "1000000"))
    p_re = float(os.environ.get("P_RE", "0.01"))
    seed = int(os.environ.get("SEED", "2"))
    threshold_seed = int(os.environ.get("THRESHOLD_SEED", str(seed + 1000)))
    num_threads = int(
        os.environ.get("NUM_THREADS", str(min(8, os.cpu_count() or 1)))
    )
    chunk_size = int(os.environ.get("CHUNK_SIZE", "2000"))
    force_thresholds = os.environ.get("FORCE_RECALIBRATE_THRESHOLDS", "0") == "1"

    output_path = Path(
        os.environ.get(
            "OUTPUT_RDATA",
            str(workdir / f"real_data_accelerated_p{p_re:g}_B{num_rep}.Rdata"),
        )
    )
    threshold_cache = Path(
        os.environ.get(
            "THRESHOLD_CACHE",
            str(
                workdir
                / f"sre_thresholds_p{p_re:g}_draws{threshold_draws}_seed{threshold_seed}.npz"
            ),
        )
    )

    set_num_threads(num_threads)
    Y0, Y1, X, B, K, H, nh, nh1, pi, starts = load_data(data_path)
    if not np.allclose(Y0, Y1):
        raise ValueError("The input file does not satisfy the sharp null Y0 == Y1.")
    Y = Y0.copy()

    print(
        f"N={Y.size}, H={H}, K={K}, NUM_REP={num_rep}, "
        f"P_RE={p_re}, threads={num_threads}",
        flush=True,
    )

    # Threshold calibration is done once and cached.  These thresholds reproduce
    # the finite-sample calibration in the original R program: p_re is the
    # acceptance probability in each stratum for SS-ReM/SS-ReP and the global
    # acceptance probability for FE-ReP.
    threshold_start = time.perf_counter()
    use_cache = False
    if threshold_cache.exists() and not force_thresholds:
        cached = np.load(threshold_cache)
        if (
            int(cached["threshold_draws"][0]) == threshold_draws
            and abs(float(cached["p_re"][0]) - p_re) < 1e-15
            and int(cached["threshold_seed"][0]) == threshold_seed
        ):
            a_rem = cached["a_rem_ss"]
            t_rep_ss = cached["t_threshold_rep_ss"]
            t_rep_mn = float(cached["t_threshold_rep_mn"][0])
            use_cache = True

    if not use_cache:
        a_rem, t_rep_ss, t_rep_mn = calibrate_thresholds(
            X=X,
            nh=nh,
            nh1=nh1,
            pi=pi,
            starts=starts,
            p_re=p_re,
            draws=threshold_draws,
            seed=threshold_seed,
            batch_size=5000,
        )
        np.savez_compressed(
            threshold_cache,
            a_rem_ss=a_rem,
            t_threshold_rep_ss=t_rep_ss,
            t_threshold_rep_mn=np.array([t_rep_mn]),
            alpha_t_ss=2.0 * norm.sf(t_rep_ss),
            alpha_t_mn=np.array([2.0 * norm.sf(t_rep_mn)]),
            p_re=np.array([p_re]),
            threshold_draws=np.array([threshold_draws]),
            threshold_seed=np.array([threshold_seed]),
        )
    threshold_seconds = time.perf_counter() - threshold_start
    print(
        f"Thresholds: {threshold_seconds:.3f} seconds "
        f"({'cache' if use_cache else 'calibrated'})",
        flush=True,
    )

    # Static regression quantities.
    subset_sizes, subset_indices = build_subsets(K)
    total_x, total_xx, total_y, total_xy = build_ss_totals(
        X, Y, nh, starts
    )
    Cpad, invpad, yres, ranks = build_mn_controls(
        X, Y, B, H, subset_sizes, subset_indices
    )
    invcovs = np.zeros((H, K, K), dtype=np.float64)
    for h in range(H):
        s0 = starts[h]
        invcovs[h] = np.linalg.pinv(
            np.cov(X[s0 : s0 + nh[h]], rowvar=False, ddof=1),
            rcond=1e-12,
        )

    # Compile all Numba kernels before timing the Monte Carlo loop.
    warmup_start = time.perf_counter()
    warmup_seed = np.array([np.uint64(seed + 987654321)], dtype=np.uint64)
    z_sre = generate_sre_numba(warmup_seed, nh, nh1, starts, Y.size)
    z_rem, _ = generate_rem_ss_numba(
        warmup_seed, X, nh, nh1, starts, invcovs, a_rem
    )
    z_rep_ss, _ = generate_rep_ss_numba(
        warmup_seed, X, nh, nh1, starts, t_rep_ss
    )
    z_rep_mn, _ = generate_rep_mn_numba(
        warmup_seed, X, nh, nh1, pi, starts, t_rep_mn
    )
    _ = analyze_ss_joint_batch(
        z_sre,
        X,
        Y,
        nh,
        nh1,
        pi,
        starts,
        subset_sizes,
        subset_indices,
        total_x,
        total_xx,
        total_y,
        total_xy,
    )
    _ = analyze_ss_joint_batch(
        z_rem,
        X,
        Y,
        nh,
        nh1,
        pi,
        starts,
        subset_sizes,
        subset_indices,
        total_x,
        total_xx,
        total_y,
        total_xy,
    )
    _ = analyze_ss_joint_batch(
        z_rep_ss,
        X,
        Y,
        nh,
        nh1,
        pi,
        starts,
        subset_sizes,
        subset_indices,
        total_x,
        total_xx,
        total_y,
        total_xy,
    )
    _ = analyze_mn_batch(
        z_sre,
        X,
        Y,
        subset_sizes,
        subset_indices,
        Cpad,
        invpad,
        yres,
        ranks,
        nh1,
    )
    _ = analyze_mn_batch(
        z_rep_mn,
        X,
        Y,
        subset_sizes,
        subset_indices,
        Cpad,
        invpad,
        yres,
        ranks,
        nh1,
    )
    warmup_seconds = time.perf_counter() - warmup_start
    print(f"Numba warm-up: {warmup_seconds:.3f} seconds", flush=True)

    hacked_p_vec_rem_ss = np.empty(num_rep, dtype=np.float64)
    hacked_p_vec_cre_ss = np.empty(num_rep, dtype=np.float64)
    hacked_p_vec_rep_ss = np.empty(num_rep, dtype=np.float64)
    hacked_p_vec_rep_mn = np.empty(num_rep, dtype=np.float64)
    hacked_p_vec_cre_mn = np.empty(num_rep, dtype=np.float64)
    tries_rem_ss = np.empty(num_rep, dtype=np.int64)
    tries_rep_ss = np.empty(num_rep, dtype=np.int64)
    tries_rep_mn = np.empty(num_rep, dtype=np.int64)

    seed_rng = np.random.default_rng(seed)
    all_seeds = seed_rng.integers(
        1, 2**63 - 1, size=num_rep, dtype=np.uint64
    )

    simulation_start = time.perf_counter()
    for lo in range(0, num_rep, chunk_size):
        hi = min(lo + chunk_size, num_rep)
        seeds_chunk = all_seeds[lo:hi]

        # Base stratified randomization.  The same assignments are used for the
        # stratum-specific and fixed-effects analyses, as in the original code.
        z_sre = generate_sre_numba(
            seeds_chunk, nh, nh1, starts, Y.size
        )
        hacked_p_vec_cre_ss[lo:hi] = analyze_ss_joint_batch(
            z_sre,
            X,
            Y,
            nh,
            nh1,
            pi,
            starts,
            subset_sizes,
            subset_indices,
            total_x,
            total_xx,
            total_y,
            total_xy,
        )
        hacked_p_vec_cre_mn[lo:hi] = analyze_mn_batch(
            z_sre,
            X,
            Y,
            subset_sizes,
            subset_indices,
            Cpad,
            invpad,
            yres,
            ranks,
            nh1,
        )

        z_rem, attempts = generate_rem_ss_numba(
            seeds_chunk, X, nh, nh1, starts, invcovs, a_rem
        )
        tries_rem_ss[lo:hi] = attempts
        hacked_p_vec_rem_ss[lo:hi] = analyze_ss_joint_batch(
            z_rem,
            X,
            Y,
            nh,
            nh1,
            pi,
            starts,
            subset_sizes,
            subset_indices,
            total_x,
            total_xx,
            total_y,
            total_xy,
        )

        z_rep_ss, attempts = generate_rep_ss_numba(
            seeds_chunk, X, nh, nh1, starts, t_rep_ss
        )
        tries_rep_ss[lo:hi] = attempts
        hacked_p_vec_rep_ss[lo:hi] = analyze_ss_joint_batch(
            z_rep_ss,
            X,
            Y,
            nh,
            nh1,
            pi,
            starts,
            subset_sizes,
            subset_indices,
            total_x,
            total_xx,
            total_y,
            total_xy,
        )

        z_rep_mn, attempts = generate_rep_mn_numba(
            seeds_chunk, X, nh, nh1, pi, starts, t_rep_mn
        )
        tries_rep_mn[lo:hi] = attempts
        hacked_p_vec_rep_mn[lo:hi] = analyze_mn_batch(
            z_rep_mn,
            X,
            Y,
            subset_sizes,
            subset_indices,
            Cpad,
            invpad,
            yres,
            ranks,
            nh1,
        )

        elapsed = time.perf_counter() - simulation_start
        done = hi
        eta = elapsed / done * (num_rep - done) if done else float("nan")
        print(
            f"Completed {done:,}/{num_rep:,}; elapsed={elapsed:.1f}s; "
            f"ETA={eta:.1f}s",
            flush=True,
        )

    simulation_seconds = time.perf_counter() - simulation_start

    method_names = np.array(
        [
            "SS-ReM",
            "SRE (SS analysis)",
            "SS-ReP",
            "FE-ReP",
            "SRE (FE analysis)",
        ],
        dtype=object,
    )
    p_vectors = [
        hacked_p_vec_rem_ss,
        hacked_p_vec_cre_ss,
        hacked_p_vec_rep_ss,
        hacked_p_vec_rep_mn,
        hacked_p_vec_cre_mn,
    ]
    type1 = np.array([np.mean(v <= 0.05) for v in p_vectors])
    mcse = np.sqrt(type1 * (1.0 - type1) / num_rep)

    output = {
        # Original object names, retained for drop-in compatibility with R.
        "hacked_p_vec_rem_ss": hacked_p_vec_rem_ss,
        "hacked_p_vec_cre_ss": hacked_p_vec_cre_ss,
        "hacked_p_vec_rep_ss": hacked_p_vec_rep_ss,
        "hacked_p_vec_rep_mn": hacked_p_vec_rep_mn,
        "hacked_p_vec_cre_mn": hacked_p_vec_cre_mn,
        # Thresholds and diagnostics.
        "a_rem_ss": a_rem,
        "alpha_t_ss": 2.0 * norm.sf(t_rep_ss),
        "alpha_t_mn": np.array([2.0 * norm.sf(t_rep_mn)]),
        "t_threshold_rep_ss": t_rep_ss,
        "t_threshold_rep_mn": np.array([t_rep_mn]),
        "tries_rem_ss": tries_rem_ss,
        "tries_rep_ss": tries_rep_ss,
        "tries_rep_mn": tries_rep_mn,
        "method_names": method_names,
        "type_I_error": type1,
        "mcse": mcse,
        "p_re": np.array([p_re]),
        "alpha": np.array([0.05]),
        "num_rep": np.array([num_rep]),
        "threshold_draws": np.array([threshold_draws]),
        "num_threads": np.array([num_threads]),
        "threshold_seconds": np.array([threshold_seconds]),
        "warmup_seconds": np.array([warmup_seconds]),
        "simulation_seconds": np.array([simulation_seconds]),
        "total_seconds": np.array(
            [threshold_seconds + warmup_seconds + simulation_seconds]
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rdata.write_rda(output_path, output, compression="gzip")

    timing_path = output_path.with_suffix(".timing.json")
    timing = {
        "N": int(Y.size),
        "H": int(H),
        "K": int(K),
        "num_rep": num_rep,
        "p_re": p_re,
        "threshold_draws": threshold_draws,
        "num_threads": num_threads,
        "threshold_cache_used": use_cache,
        "threshold_seconds": threshold_seconds,
        "warmup_seconds": warmup_seconds,
        "simulation_seconds": simulation_seconds,
        "seconds_per_replication": simulation_seconds / num_rep,
        "total_seconds": threshold_seconds + warmup_seconds + simulation_seconds,
        "mean_attempts_per_final_assignment": {
            "SS-ReM_stratum_draws_total": float(np.mean(tries_rem_ss)),
            "SS-ReP_stratum_draws_total": float(np.mean(tries_rep_ss)),
            "FE-ReP_whole_assignments": float(np.mean(tries_rep_mn)),
        },
        "type_I_error": {
            str(method_names[j]): float(type1[j])
            for j in range(method_names.size)
        },
        "mcse": {
            str(method_names[j]): float(mcse[j])
            for j in range(method_names.size)
        },
    }
    timing_path.write_text(json.dumps(timing, indent=2), encoding="utf-8")

    print(f"Simulation: {simulation_seconds:.3f} seconds", flush=True)
    print(f"Output RData: {output_path}", flush=True)
    print(f"Timing JSON: {timing_path}", flush=True)
    for j, name in enumerate(method_names):
        print(
            f"{name}: Type I error={type1[j]:.4f}, MCSE={mcse[j]:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main_accelerated()
