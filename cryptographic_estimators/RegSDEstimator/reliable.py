"""Complete searches for the explicitly balanced ES24 binary concrete models.

This certifies optimization of the stated finite grid, not cryptographic security
or completeness of the attack literature. The two block halves use floor(w/2) and ceil(w/2) separately; no equal-size
rounding is silently imposed for odd w.
"""
from functools import lru_cache
from math import comb, log2, floor, isfinite
import numpy as np
from ..search_contract import SearchBudget, SearchIncomplete, POLICY_VERSION


@lru_cache(maxsize=32768)
def logcomb(n, r):
    return log2(comb(n, r)) if 0 <= r <= n else -np.inf


def representation_count_log(w,p,eps,v):
    """ES24 balanced representations, including the positions of cancellations."""
    return (2*logcomb(p//2,p//4)+logcomb(w//2-p//2,eps//2)
            +logcomb((w+1)//2-p//2,eps//2)+eps*np.log2(v))


def representation_floor(w,p,eps,v_numerator):
    """Floor log2(R) by exact integer comparisons at rational v=num/w."""
    numerator=(comb(p//2,p//4)**2*comb(w//2-p//2,eps//2)
               *comb((w+1)//2-p//2,eps//2)*int(v_numerator)**eps)
    denominator=w**eps
    exponent=numerator.bit_length()-denominator.bit_length()
    below = numerator < (denominator << exponent) if exponent>=0 else (numerator << -exponent) < denominator
    if below:
        exponent-=1
    return exponent


def representation_floor_vector(w,p,eps,numerators):
    if eps==0:
        return np.full(len(numerators),representation_floor(w,p,eps,1),dtype=float)
    logs=representation_count_log(w,p,eps,numerators/w)
    result=np.floor(logs)
    for i in np.flatnonzero(np.abs(logs-np.rint(logs))<1e-9):
        result[i]=representation_floor(w,p,eps,int(numerators[i]))
    return result


def balanced_point(n,k,w,parameters,depth=3,representation_model="es24"):
    """Scalar cost oracle, separately evaluated from the vectorized optimizer."""
    p,ell=parameters['p'],parameters['ell']
    ex,ey=parameters.get('eps_x',0),parameters.get('eps_y',0)
    h0,h1=w//2,(w+1)//2
    v=(k-w+ell)/w;b=n/w
    if not (0<v<b and 0<=p<=2*h0 and p%2==0):return np.inf,np.inf
    lv=log2(v);logn=log2(n);gauss=2*log2(n-k+w)
    prob=logcomb(h0,p//2)+logcomb(h1,p//2)+p*log2(v/b)+(w-p)*log2(1-v/b)
    def add(a,b):
        m=max(a,b)
        return m+log2(2**(a-m)+2**(b-m))
    if depth==1:
        L=logcomb(h0,p//2)+lv*(p//2);R=logcomb(h1,p//2)+lv*(p//2)
        return logn+max(gauss,add(L,R),L+R-ell)-prob,logn+max(L,R)
    px=p//2+ex
    if p%4 or ex%2 or ex<0 or ex>w-p:return np.inf,np.inf
    lx=representation_floor(w,p,ex,k-w+ell if representation_model=='es24' else w)
    if lx>ell:return np.inf,np.inf
    if depth==2:
        L=logcomb(h0,px//2)+lv*(px//2);R=logcomb(h1,px//2)+lv*(px//2)
        Lx=L+R-lx;Nx=2*Lx-ell+lx
        return logn+max(gauss,1+add(L,R),1+Lx,Nx)-prob,logn+max(L,R,Lx)
    if px%4 or ey%2 or ey<0 or ey>w-px:return np.inf,np.inf
    py=px//2+ey
    ly=representation_floor(w,px,ey,k-w+ell)
    if ly>lx:return np.inf,np.inf
    L=logcomb(h0,py//2)+lv*(py//2);R=logcomb(h1,py//2)+lv*(py//2)
    Ly=L+R-ly;Ny=2*Ly-lx+ly
    Lx=logcomb(h0,px//2)+logcomb(h1,px//2)+lv*px-lx
    Nx=2*Lx-ell+lx
    return logn+max(gauss,2+add(L,R),2+Ly,1+Ny,1+Lx,Nx)-prob,logn+max(L,R,Ly,Lx)


def balanced_regular(n, k, w, depth=3, max_evaluations=None, max_seconds=None,
                     representation_model="es24"):
    """depth=1: Enum; depth=2/3: Rep. All costs include log2(n) once.

    Safe pruning uses an actual term of T_iter / Pr[success], pointwise in ell.
    There are no interior-point or unimodality assumptions and no epsilon caps.
    Non-integral v follows ES24 Appendix A.1's explicitly unrounded model.
    """
    domain = dict(n=n,k=k,w=w,depth=depth,ell=[0,n-k+w-1],
                  p_step=2 if depth==1 else 4,eps_x_step=2,eps_y_step=2,
                  constraints="p<=w, eps_x<=w-p; depth3: p_x%4=0, eps_y<=w-p_x",
                  rounding="floor/ceil block halves, integer balanced list weights; continuous v",
                  representation_model=representation_model)
    budget=SearchBudget("RegularISD"+str(depth),domain,max_evaluations,max_seconds)
    if depth not in (1,2,3) or not (0<w<k<n and n%w==0):
        raise SearchIncomplete("Unsupported balanced-model instance",budget.report("unsupported_model"))
    if representation_model not in ("es24","legacy_lpn_depth2") or (representation_model!="es24" and depth!=2):
        raise ValueError("Unknown representation model")
    ell=np.arange(n-k+w)
    v=(k-w+ell)/w
    lv=np.log2(v); a=np.log2(v/(n//w)); b=np.log2(1-v/(n//w))
    gauss=2*log2(n-k+w); logn=log2(n)
    h0,h1=w//2,(w+1)//2
    # A safe initial incumbent: the p=0 permutation attack in the same model.
    budget.consider(logn+max(gauss,depth)-w*log2(1-(k-w)/n),(dict(p=0,ell=0) if depth==1 else dict(p=0,ell=0,eps_x=0) if depth==2 else dict(p=0,ell=0,eps_x=0,eps_y=0)),logn)

    @lru_cache(maxsize=None)
    def base_lists(weight):
        left=logcomb(h0,weight//2)+lv*(weight//2)
        right=logcomb(h1,weight//2)+lv*(weight//2)
        initial=2+np.logaddexp2(left,right)
        return left,right,initial,logn+np.maximum(gauss,initial)

    def take(costs, memory, pars, valid=None):
        if valid is not None:
            costs=np.where(valid,costs,np.inf)
        if np.isnan(costs).any():
            raise SearchIncomplete("NaN in cost vector",budget.report("numerical_failure"))
        idx=int(np.argmin(costs))
        if isfinite(costs[idx]):
            budget.consider(float(costs[idx]),dict(pars,ell=int(ell[idx])),float(memory[idx])+logn)

    for p in range(0,w+1,2 if depth==1 else 4):
        prob=logcomb(h0,p//2)+logcomb(h1,p//2)+a*p+b*(w-p)
        allowed_base_weights={}
        if depth==1:
            budget.tick(len(ell))
            L=logcomb(h0,p//2)+lv*(p//2)
            R=logcomb(h1,p//2)+lv*(p//2)
            take(logn+np.maximum(np.maximum(gauss,np.logaddexp2(L,R)),L+R-ell)-prob,np.maximum(L,R),dict(p=p))
            continue
        for ex in range(0,w-p+1,2):
            budget.tick()  # also check time limits in pruned branches
            px=p//2+ex
            if depth==3 and px%4:continue
            lx=(representation_floor_vector(w,p,ex,k-w+ell) if representation_model=="es24"
                else np.full_like(v,representation_floor(w,p,ex,w)))
            if np.ndim(lx)==0:lx=np.full_like(v,lx)
            if depth==2:
                L=logcomb(h0,px//2)+lv*(px//2)
                R=logcomb(h1,px//2)+lv*(px//2)
                initial=1+np.logaddexp2(L,R)
                lower=logn+np.maximum(gauss,initial)-prob
                if float(np.min(lower))>budget.best['bits']+1e-9:continue
                budget.tick(len(ell))
                Lx=L+R-lx;Nx=2*Lx-(ell-lx)
                take(logn+np.maximum.reduce([np.full_like(v,gauss),initial,1+Lx,Nx])-prob,
                     np.maximum(np.maximum(L,R),Lx),dict(p=p,eps_x=ex),lx<=ell)
            else:
                for ey in range(0,w-px+1,2):
                    budget.tick()
                    py=px//2+ey
                    L,R,initial,base_lower=base_lists(py)
                    if py not in allowed_base_weights:
                        allowed_base_weights[py]=float(np.min(base_lower-prob))<=budget.best['bits']+1e-9
                    # An earlier True remains conservative as the incumbent drops;
                    # an earlier False cannot become feasible. No shape assumption.
                    if not allowed_base_weights[py]:continue
                    budget.tick(len(ell))
                    ly=representation_floor_vector(w,px,ey,k-w+ell)
                    Ly=L+R-ly;Ny=2*Ly-(lx-ly)
                    Lx=logcomb(h0,px//2)+logcomb(h1,px//2)+lv*px-lx;Nx=2*Lx-(ell-lx)
                    take(logn+np.maximum.reduce([np.full_like(v,gauss),initial,2+Ly,1+Ny,1+Lx,Nx])-prob,
                         np.maximum.reduce([L,R,Ly,Lx]),dict(p=p,eps_x=ex,eps_y=ey),
                         (ly<=lx)&(lx<=ell))
    return budget.finish()


def estimate(n,k,w,max_evaluations=None,max_seconds=None,include_sd=True):
    """Strict portfolio: an incomplete component prevents acceptance of the minimum."""
    from .regsd_problem import RegSDProblem
    from .RegSDAlgorithms import RegularISDPerm, CCJLin, CCJ, SDAttack
    problem=RegSDProblem(n,k,w)
    reports=[]
    for cls in (RegularISDPerm,CCJLin):
        algo=cls(problem,bit_complexities=1,memory_access=0)
        budget=SearchBudget(cls.__name__,{"optimization":"closed formula"})
        budget.consider(algo.time_complexity(),{})
        reports.append(budget.finish())
    for depth in (1,3):
        reports.append(balanced_regular(n,k,w,depth,max_evaluations,max_seconds))
    ccj=CCJ(problem,bit_complexities=1,memory_access=0)
    budget=SearchBudget("CCJ",dict(ell=[0,n-k]),max_evaluations,max_seconds)
    for ell in range(n-k+1):
        budget.tick();value=ccj._compute_time_and_memory_complexity({'ell':ell})[0]+log2(n)
        if isfinite(value):budget.consider(value,{'ell':ell})
    reports.append(budget.finish())
    if include_sd:
        sd=SDAttack(problem,bit_complexities=1,memory_access=0,
                    full_domain=True,max_evaluations=max_evaluations,max_seconds=max_seconds)
        for algorithm in sd.SDEstimator.algorithms():
            value=algorithm.time_complexity()+log2(n)
            report=getattr(algorithm,"search_report",None)
            if report is None and algorithm.__class__.__name__=="Prange":
                report={"status":"complete","algorithm":"Prange","domain":{"r":"all M4RI blocksizes; elimination objective"}}
            if report is None:
                raise RuntimeError("Missing search completion report")
            if report['status']!='complete':
                raise SearchIncomplete("SD search not complete",report)
            report=dict(report,best=dict(bits=value,parameters=algorithm.optimal_parameters()))
            reports.append(report)
    best=min(reports,key=lambda r:r['best']['bits'])
    return dict(bits=best['best']['bits'],status="complete",policy_version=POLICY_VERSION,
                algorithms=reports,best_algorithm=best['algorithm'],
                coverage="ES24 Perm/Enum/Rep3, CCJ/CCJLin"+("; upstream SDAttack portfolio" if include_sd else "; SD explicitly excluded"))
