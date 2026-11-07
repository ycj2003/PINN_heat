"""Paper-aligned *analytic subcase*, paired soft/hard PINNs.

The source is Bowman et al. (2023), Algorithms 16, 428, Eq. (12):
  u*(x,t)=exp(-alpha*pi*pi*t)*sin(pi*x),    u_t-alpha*u_xx=0.
We enforce the Dirichlet temperature traces of that analytic function.
IMPORTANT: this is NOT the paper's full PAC1D insulated/convection/laser experiment.
Original PAC1D data were not publicly released at publication.

Paper-aligned components: 10x32 Tanh layers, 21:1 condition/PDE loss ratio,
25,000-point LHS candidate pool, 125 IC points, 175 BC points, L-BFGS stage.
Practical deviations: alpha=.1 nondimensional, stochastic Adam warmup, minibatch
PDE updates, LBFGS on 1024 fixed collocations rather than 25,000 full-batch.
"""
from __future__ import annotations
import csv, json, math, random, time, argparse
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scipy.stats import qmc
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results' / 'paper_aligned'
OUT.mkdir(parents=True, exist_ok=True)
ALPHA = 0.1
ARCH = (2, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 1)
SEED = 11
DEVICE = 'cpu'
ADAM_EPOCHS = 2000
N_POOL = 25000
N_PER_STEP = 256
N_IC = 125
N_BC = 175
LBFGS_STEPS = 120
LBFGS_POINTS = 1024
ICBC_WEIGHT = 21.0
PDE_WEIGHT = 1.0


def exact(x, t):
    return torch.exp(-ALPHA * math.pi**2 * t) * torch.sin(math.pi*x)

def initial(x):
    return torch.sin(math.pi*x)

class Core(nn.Module):
    def __init__(self):
        super().__init__()
        layers=[]
        for i in range(len(ARCH)-1):
            layers.append(nn.Linear(ARCH[i], ARCH[i+1]))
            if i < len(ARCH)-2:
                layers.append(nn.Tanh())
        self.net=nn.Sequential(*layers)
    def forward(self,xt): return self.net(xt)

class PINN(nn.Module):
    def __init__(self,mode):
        super().__init__()
        self.mode=mode
        self.core=Core()
    def forward(self,xt):
        raw=self.core(xt)
        if self.mode=='soft': return raw
        x,t=xt[:,:1],xt[:,1:]
        return initial(x)+t*x*(1-x)*raw


def physics_loss(model, xt):
    coords=xt.detach().requires_grad_(True)
    u=model(coords)
    g=torch.autograd.grad(u,coords,torch.ones_like(u),create_graph=True)[0]
    g2=torch.autograd.grad(g[:,:1],coords,torch.ones_like(g[:,:1]),create_graph=True)[0]
    r=g[:,1:]-ALPHA*g2[:,:1]
    return r.square().mean()


def condition_loss(model,ic,bc):
    a=model(torch.cat([ic,torch.zeros_like(ic)],dim=1))-initial(ic)
    zz=torch.zeros_like(bc)
    l=model(torch.cat([zz,bc],dim=1))
    r=model(torch.cat([zz+1,bc],dim=1))
    return a.square().mean(), 0.5*(l.square().mean()+r.square().mean())


def eval_metrics(model,mode,seed,train_seconds):
    grid=torch.linspace(0,1,101)
    xx,tt=torch.meshgrid(grid,grid,indexing='ij')
    with torch.no_grad():
        pred=model(torch.stack([xx.ravel(),tt.ravel()],1)).reshape(101,101)
        target=exact(xx,tt)
    diff=pred-target
    rel=float(diff.norm()/target.norm())
    rmse=float(torch.sqrt(diff.square().mean()))
    r2=1-float(diff.square().sum() / (target-target.mean()).square().sum())
    ic=float(torch.sqrt((pred[:,0]-target[:,0]).square().mean()))
    bc=float(torch.sqrt(0.5*(pred[0,:].square().mean()+pred[-1,:].square().mean())))
    # Held-out PDE evaluation independent from the 25k LHS candidate pool.
    heldout=torch.from_numpy(qmc.LatinHypercube(d=2,seed=84211).random(2048)).float()
    rmse_pde=float(physics_loss(model,heldout).sqrt().detach())
    out=dict(benchmark='Bowman-2023-Eq12-analytic-no-source-subcase',
             mode=mode, seed=seed, alpha=ALPHA, architecture=list(ARCH),
             adam_epochs=ADAM_EPOCHS, lbfgs_steps=LBFGS_STEPS,
             n_lhs_candidate_pool=N_POOL, n_adam_collocation_per_step=N_PER_STEP,
             n_lbfgs_fixed_collocation=LBFGS_POINTS,
             n_ic=N_IC,n_bc=N_BC,ic_bc_weight=ICBC_WEIGHT,pde_weight=PDE_WEIGHT,
             relative_l2=rel,rmse=rmse,r2=r2,ic_rmse=ic,
             bc_rmse=bc,pde_rmse_heldout=rmse_pde,training_seconds=train_seconds)
    return out, pred.detach().numpy(),target.detach().numpy()


def draw_predictions(preds, truth):
    # Pillow only - safe on Windows environments where Matplotlib restarts the kernel.
    w,h=1160,430
    im=Image.new('RGB',(w,h),'white'); draw=ImageDraw.Draw(im)
    def font(size):
        for name in ('DejaVuSans.ttf','arial.ttf'):
            try:return ImageFont.truetype(name,size)
            except:pass
        return ImageFont.load_default()
    f=font(20)
    arrays=[truth,preds['soft'],preds['hard'],abs(preds['soft']-truth),abs(preds['hard']-truth)]
    titles=['Exact Eq.(12)','Soft prediction','Hard prediction','Soft absolute error','Hard absolute error']
    from PIL import ImageOps
    for j,(a,title) in enumerate(zip(arrays,titles)):
        # image indexes x down / t right; orient display with origin at lower-left
        lo,hi=((-1,1) if j<=2 else (0,max(float(arrays[3].max()),float(arrays[4].max()),1e-9)))
        # for exact + prediction use common 0..1 consistent color scale
        if j<=2:lo,hi=(0,1)
        gray=np.clip((a.T[::-1]-lo)/(hi-lo),0,1)
        r=(35+220*gray).astype(np.uint8);g=(57+120*(1-abs(gray-.5)*2)).astype(np.uint8);b=(195-160*gray).astype(np.uint8)
        tile=Image.fromarray(np.stack([r,g,b],-1),'RGB').resize((208,290),Image.Resampling.NEAREST)
        x=20+j*229
        im.paste(tile,(x,66))
        draw.text((x,25),title,fill='#1c2630',font=f)
        draw.text((x,370),'x: left-right; t: up-down',fill='#64748b',font=font(13))
    im.save(OUT/'paper_aligned_predictions.png')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');parser.add_argument('--mode',choices=['soft','hard','both'],default='both')
    args=parser.parse_args()
    torch.set_num_threads(min(4,torch.get_num_threads()))
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
    lhs=torch.from_numpy(qmc.LatinHypercube(d=2,seed=SEED+200).random(N_POOL)).float()
    generator=torch.Generator().manual_seed(SEED+9)
    # Same IC/BC points for soft, and same initial core weights for soft/hard.
    ic=torch.rand(N_IC,1,generator=generator)
    bc=torch.rand(N_BC,1,generator=generator)
    fixed=lhs[:LBFGS_POINTS].clone()
    steps=30 if args.smoke else ADAM_EPOCHS
    lbfgs_steps=2 if args.smoke else LBFGS_STEPS
    all_metrics={};preds={};histories={}
    for mode in (('soft','hard') if args.mode=='both' else (args.mode,)):
        torch.manual_seed(SEED)
        model=PINN(mode)
        torch.manual_seed(SEED+310)  # matched PDE samples across paired runs
        sample_gen=torch.Generator().manual_seed(SEED+310)
        optim=torch.optim.Adam(model.parameters(),lr=1e-3)
        history=[]
        t0=time.perf_counter()
        print(f'RUN {mode}: Adam steps={steps}, LBFGS maximum steps={lbfgs_steps}',flush=True)
        for epoch in range(1,steps+1):
            idx=torch.randint(0,N_POOL,(N_PER_STEP,),generator=sample_gen)
            f=physics_loss(model,lhs[idx])
            if mode=='soft':
                li,lb=condition_loss(model,ic,bc)
                loss=ICBC_WEIGHT*(li+lb)+PDE_WEIGHT*f
            else:
                li=lb=torch.zeros(())
                loss=PDE_WEIGHT*f
            optim.zero_grad(set_to_none=True); loss.backward();optim.step()
            history.append(dict(stage='Adam',step=epoch,total=float(loss.detach()),physics=float(f.detach()),ic=float(li.detach()),bc=float(lb.detach())))
            if epoch%500==0 or epoch==steps:
                print(f' {mode} adam {epoch}/{steps}: total={float(loss.detach()):.4g} pde={float(f.detach()):.4g}',flush=True)
        # Paper uses L-BFGS; do a modest deterministic refinement under fixed subset.
        optimizer2=torch.optim.LBFGS(model.parameters(),lr=.7,max_iter=lbfgs_steps,
                                      tolerance_grad=1e-9,tolerance_change=1e-12,
                                      history_size=35,line_search_fn='strong_wolfe')
        closure_calls=[0]
        def closure():
            optimizer2.zero_grad(set_to_none=True)
            f=physics_loss(model,fixed)
            if mode=='soft':li,lb=condition_loss(model,ic,bc);loss=ICBC_WEIGHT*(li+lb)+f
            else:li=lb=torch.zeros(());loss=f
            loss.backward()
            closure_calls[0]+=1
            history.append(dict(stage='LBFGS',step=closure_calls[0],total=float(loss.detach()),physics=float(f.detach()),ic=float(li.detach()),bc=float(lb.detach())))
            return loss
        optimizer2.step(closure)
        elapsed=time.perf_counter()-t0
        with (OUT/f'{mode}_history.csv').open('w',newline='',encoding='utf8') as file:
            writer=csv.DictWriter(file,fieldnames=['stage','step','total','physics','ic','bc'])
            writer.writeheader();writer.writerows(history)
        if not args.smoke:
            torch.save(model.state_dict(),OUT/f'{mode}_model.pt')
        met,pred,truth=eval_metrics(model,mode,SEED,elapsed)
        met['lbfgs_closure_calls']=closure_calls[0]
        all_metrics[mode]=met;preds[mode]=pred;histories[mode]=history
        (OUT/f'{mode}_metrics.json').write_text(json.dumps(met,indent=2),encoding='utf8')
        print('RESULT',mode,'relative_L2=',met['relative_l2'],'PDE_RMSE=',met['pde_rmse_heldout'],'R2=',met['r2'],'elapsed=',elapsed,flush=True)
        del model
    if not args.smoke:
        for mode in ('soft','hard'):
            if mode not in all_metrics and (OUT/f'{mode}_model.pt').exists():
                mdl=PINN(mode)
                mdl.load_state_dict(torch.load(OUT/f'{mode}_model.pt',map_location='cpu',weights_only=True))
                met,pred,_=eval_metrics(mdl,mode,SEED,float('nan'))
                existing=OUT/f'{mode}_metrics.json'
                if existing.exists():met=json.loads(existing.read_text(encoding='utf8'))
                all_metrics[mode]=met;preds[mode]=pred
        (OUT/'metrics.json').write_text(json.dumps(all_metrics,indent=2),encoding='utf8')
        if len(preds)==2:draw_predictions(preds,truth)
        np.savetxt(ROOT/'data'/'paper_eq12_reference_101x101.csv',
                   np.column_stack([torch.linspace(0,1,101).repeat_interleave(101).numpy(),
                                    torch.linspace(0,1,101).repeat(101).numpy(),truth.ravel()]),
                   header='x,t,u_exact',delimiter=',',comments='')
    print('DONE',flush=True)

if __name__=='__main__':main()
