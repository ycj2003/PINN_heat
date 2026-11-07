"""Five paired seeds (1-5) for soft/hard PINNs on the manufactured heat equation.

From the repository root, run: python run_multiseed.py
No external data or Matplotlib is required. All scores are computed on the same
held-out evaluation sets. Do not use the reference solution for training.

This reproduces the formulas, architecture, loss and evaluation of pinn_heat.py;
only the seeds and output paths are changed to support replicate runs.
"""
from __future__ import annotations

import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results' / 'multi_seed'
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = (1, 2, 3, 4, 5)
ALPHA, EPOCHS, N_F, N_IC, N_BC = 0.1, 2000, 256, 128, 128
LR, LAMBDA_IC, LAMBDA_BC = 1e-3, 10.0, 10.0
GRID_SIZE = 101
DEVICE = torch.device('cpu')
torch.set_num_threads(min(4, torch.get_num_threads()))


def initial_u(x):
    return torch.sin(math.pi * x)


def exact_u(x, t):
    return torch.exp(-t)*torch.sin(math.pi*x) + 0.2*t*torch.sin(3*math.pi*x)


def heat_source(x, t):
    return ((ALPHA*math.pi**2 - 1)*torch.exp(-t)*torch.sin(math.pi*x)
            + 0.2*(1 + 9*ALPHA*math.pi**2*t)*torch.sin(3*math.pi*x))


class HeatMLP(nn.Module):
    def __init__(self, width=64, depth=3):
        super().__init__()
        layers = [nn.Linear(2, width), nn.Tanh()]
        for _ in range(depth-1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, xt):
        return self.net(xt)


class HeatPINN(nn.Module):
    def __init__(self, mode):
        super().__init__()
        assert mode in ('soft', 'hard')
        self.mode = mode
        self.core = HeatMLP()

    def forward(self, xt):
        raw = self.core(xt)
        if self.mode == 'soft':
            return raw
        x, t = xt[:, :1], xt[:, 1:]
        return initial_u(x) + t*x*(1-x)*raw


def pde_residual(model, x, t):
    xt = torch.cat([x, t], dim=1).detach().requires_grad_(True)
    u = model(xt)
    du = torch.autograd.grad(u, xt, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(du[:, :1], xt, torch.ones_like(u), create_graph=True)[0][:, :1]
    return du[:, 1:] - ALPHA*u_xx - heat_source(xt[:, :1], xt[:, 1:])


def soft_constraint_loss(model, generator):
    x_ic = torch.rand(N_IC, 1, generator=generator)
    t_bc = torch.rand(N_BC, 1, generator=generator)
    ic = (model(torch.cat([x_ic, torch.zeros_like(x_ic)], 1)) - initial_u(x_ic)).square().mean()
    left = model(torch.cat([torch.zeros_like(t_bc), t_bc], 1))
    right = model(torch.cat([torch.ones_like(t_bc), t_bc], 1))
    bc = 0.5*(left.square().mean() + right.square().mean())
    return ic, bc


def train_one(seed, mode):
    # Reset to the SAME raw MLP initialization in each paired mode.
    torch.manual_seed(seed)
    model = HeatPINN(mode).to(DEVICE)
    interior_rng = torch.Generator(device='cpu').manual_seed(314159 + seed)
    extra_rng = torch.Generator(device='cpu').manual_seed(271828 + seed)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    history = []
    started = time.perf_counter()
    for epoch in range(1, EPOCHS+1):
        model.train()
        xf = torch.rand(N_F, 1, generator=interior_rng)
        tf = torch.rand(N_F, 1, generator=interior_rng)
        opt.zero_grad(set_to_none=True)
        physics = pde_residual(model, xf, tf).square().mean()
        if mode == 'soft':
            ic, bc = soft_constraint_loss(model, extra_rng)
            loss = physics + LAMBDA_IC*ic + LAMBDA_BC*bc
        else:
            ic, bc = physics.detach().new_zeros(()), physics.detach().new_zeros(())
            loss = physics
        loss.backward()
        opt.step()
        history.append([epoch, float(loss.detach()), float(physics.detach()), float(ic.detach()), float(bc.detach())])
        if epoch == 1 or epoch % 500 == 0:
            print(f'seed={seed} mode={mode} epoch={epoch}/{EPOCHS} '
                  f'PDE_MSE={history[-1][2]:.3e} total={history[-1][1]:.3e}', flush=True)
    seconds = time.perf_counter()-started
    with (OUT/f'{mode}_seed{seed}_history.csv').open('w', newline='', encoding='utf8') as f:
        writer=csv.writer(f)
        writer.writerow(['epoch','total','physics','ic','bc'])
        writer.writerows(history)
    torch.save(model.state_dict(), OUT/f'{mode}_seed{seed}_model.pt')
    return model, history, seconds


def evaluate_one(seed, mode, model, seconds):
    grid = torch.linspace(0, 1, GRID_SIZE)
    xx, tt = torch.meshgrid(grid, grid, indexing='ij')
    xt = torch.stack([xx.reshape(-1), tt.reshape(-1)], dim=1)
    model.eval()
    with torch.no_grad():
        pred = model(xt).reshape(GRID_SIZE, GRID_SIZE)
        exact = exact_u(xx, tt)
        rel_l2 = ((pred-exact).norm()/exact.norm()).item()
        ic_rmse = ((pred[:, 0]-exact[:, 0]).square().mean().sqrt()).item()
        bc_rmse = (0.5*(pred[0, :].square().mean()+pred[-1, :].square().mean())).sqrt().item()
    gen = torch.Generator(device='cpu').manual_seed(123456)
    xf = torch.rand(2048, 1, generator=gen)
    tf = torch.rand(2048, 1, generator=gen)
    physics_rmse = pde_residual(model, xf, tf).detach().square().mean().sqrt().item()
    return dict(seed=seed, mode=mode, epochs=EPOCHS, relative_l2=rel_l2,
                ic_rmse=ic_rmse, bc_rmse=bc_rmse, pde_rmse_heldout=physics_rmse,
                training_seconds=seconds, device=str(DEVICE))


def load_font(size, bold=False):
    names = ['DejaVuSans-Bold.ttf', 'arialbd.ttf'] if bold else ['DejaVuSans.ttf','arial.ttf']
    for name in names:
        try:
            return ImageFont.truetype(name,size)
        except OSError:
            pass
    return ImageFont.load_default()


def plot_paired_relative_errors(rows):
    width, height = 1000, 620
    image = Image.new('RGB',(width,height),'white')
    d = ImageDraw.Draw(image)
    small, bold = load_font(20), load_font(30,True)
    soft = [100*next(r['relative_l2'] for r in rows if r['seed']==s and r['mode']=='soft') for s in SEEDS]
    hard = [100*next(r['relative_l2'] for r in rows if r['seed']==s and r['mode']=='hard') for s in SEEDS]
    ymax = max(max(soft), max(hard))*1.16
    x0, x1, y0, y1 = 95, 940, 95, 520
    for j in range(6):
        value = ymax*j/5
        y = y1-(y1-y0)*j/5
        d.line((x0,y,x1,y), fill=(220,225,235),width=1)
        d.text((12,y-12),f'{value:.2f}%',font=small,fill=(50,50,50))
    d.text((240,25),'Test relative L2 error across five seeds',font=bold,fill=(30,30,30))
    d.text((805,70),'Soft',font=small,fill=(35,91,162))
    d.text((885,70),'Hard',font=small,fill=(193,84,50))
    for i, seed in enumerate(SEEDS):
        center = x0+(i+.5)*(x1-x0)/5
        for value, offset, color in [(soft[i],-22,(35,91,162)),(hard[i],22,(193,84,50))]:
            bh = value/ymax*(y1-y0)
            d.rectangle((center+offset-17,y1-bh,center+offset+17,y1),fill=color)
        d.text((center-25,y1+15),f'Seed {seed}',font=small,fill=(45,45,45))
    image.save(OUT/'relative_l2_by_seed.png')


def plot_pde_histories(allhist):
    W,H = 1050,645
    image=Image.new('RGB',(W,H),'white')
    d=ImageDraw.Draw(image)
    small,bold=load_font(20),load_font(29,True)
    x0,x1,y0,y1=110,980,100,550
    lines=[]
    for seed in SEEDS:
        for mode in ('soft','hard'):
            hist=allhist[(seed,mode)]
            logs=np.log10(np.maximum([r[2] for r in hist],1e-15))
            lines.append((seed,mode,logs))
    lo,hi=math.floor(min(float(x.min()) for _,_,x in lines)),math.ceil(max(float(x.max()) for _,_,x in lines))
    if lo==hi: hi+=1
    for p in range(lo,hi+1):
        y=y1-(p-lo)/(hi-lo)*(y1-y0)
        d.line((x0,y,x1,y),fill=(225,225,225))
        d.text((27,y-12),f'10^{p}',font=small,fill=(65,65,65))
    for epoch in range(0,EPOCHS+1,500):
        x=x0+epoch/EPOCHS*(x1-x0)
        d.line((x,y0,x,y1),fill=(240,240,240))
        d.text((x-25,y1+12),str(epoch),font=small,fill=(65,65,65))
    d.rectangle((x0,y0,x1,y1),outline=(70,70,70),width=2)
    # Each seed in the same color family, avoid overinterpretation of one run.
    for seed,mode,logs in lines:
        color=(35+5*seed,91+3*seed,162+4*seed) if mode=='soft' else (193+4*seed,84+3*seed,50+4*seed)
        pts=[(round(x0+i/(EPOCHS-1)*(x1-x0)),round(y1-(float(v)-lo)/(hi-lo)*(y1-y0))) for i,v in enumerate(logs)]
        d.line(pts,fill=color,width=2)
    d.text((255,25),'Training PDE residual MSE: 5 seeds each',font=bold,fill=(30,30,30))
    d.text((720,65),'Soft (blue)',font=small,fill=(40,90,170))
    d.text((858,65),'Hard (orange)',font=small,fill=(190,80,50))
    d.text((520,600),'Epoch',font=small,fill=(50,50,50))
    image.save(OUT/'pde_loss_all_seeds.png')


def mean_std(values):
    return dict(mean=float(np.mean(values)), std=float(np.std(values,ddof=1)),
                min=float(np.min(values)),max=float(np.max(values)))


def main():
    print('Paired seed experiment, full runs, CPU torch',torch.__version__, flush=True)
    t0=time.perf_counter()
    # Check manufactured source using autodiff, without training labels.
    z=torch.rand(80,2,requires_grad=True)
    u=exact_u(z[:,:1],z[:,1:]);du=torch.autograd.grad(u,z,torch.ones_like(u),create_graph=True)[0]
    dxx=torch.autograd.grad(du[:,:1],z,torch.ones_like(u),create_graph=True)[0][:,:1]
    max_res=float((du[:,1:]-ALPHA*dxx-heat_source(z[:,:1],z[:,1:])).abs().max())
    assert max_res<2e-5, f'Invalid manufactured solution: {max_res}'
    rows=[]; allhist={}
    for seed in SEEDS:
        random.seed(seed);np.random.seed(seed)
        for mode in ('soft','hard'):
            model, hist, seconds = train_one(seed,mode)
            m = evaluate_one(seed,mode,model,seconds)
            rows.append(m);allhist[(seed,mode)] = hist
            print(f'RESULT seed={seed} {mode}: relL2={100*m["relative_l2"]:.5f}% '
                  f'IC={m["ic_rmse"]:.3e} BC={m["bc_rmse"]:.3e} '
                  f'PDE={m["pde_rmse_heldout"]:.3e} elapsed={seconds:.2f}s',flush=True)
            assert len(hist)==EPOCHS and np.isfinite(m['relative_l2'])
            if mode=='hard':
                assert m['ic_rmse']<1e-6 and m['bc_rmse']<1e-6
            del model
    fields=['seed','mode','epochs','relative_l2','ic_rmse','bc_rmse','pde_rmse_heldout','training_seconds','device']
    with (OUT/'per_seed_metrics.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    aggregate={}
    for mode in ('soft','hard'):
        subset=[r for r in rows if r['mode']==mode]
        aggregate[mode]={key:mean_std([r[key] for r in subset]) for key in
                        ('relative_l2','ic_rmse','bc_rmse','pde_rmse_heldout','training_seconds')}
    pairs={metric:[] for metric in ('relative_l2','pde_rmse_heldout')}
    for seed in SEEDS:
        a=next(r for r in rows if r['seed']==seed and r['mode']=='soft')
        b=next(r for r in rows if r['seed']==seed and r['mode']=='hard')
        for metric in pairs: pairs[metric].append(a[metric]-b[metric])
    paired_diffs={metric:mean_std(deltas) for metric,deltas in pairs.items()}
    summary={'benchmark':'1D manufactured heat, same as earlier seed 42',
             'seeds':list(SEEDS),'epochs_per_run':EPOCHS,'architecture':'2-64-64-64-1, Tanh',
             'alpha':ALPHA, 'lr':LR, 'n_f_per_epoch':N_F,'n_ic':N_IC, 'n_bc':N_BC,
             'lambda_ic':LAMBDA_IC,'lambda_bc':LAMBDA_BC,
             'pde_sample_seed_formula':'314159 + seed (same both modes)',
             'ic_bc_sample_seed_formula':'271828 + seed (soft only)',
             'heldout_pde_seed':123456,'grid_size':GRID_SIZE,
             'manufactured_solution_max_residual':max_res,
             'metrics':aggregate,'paired_soft_minus_hard':paired_diffs,
             'total_wall_seconds':time.perf_counter()-t0,'device':str(DEVICE),
             'torch_version':torch.__version__}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    plot_paired_relative_errors(rows)
    plot_pde_histories(allhist)
    print('\\n===== FIVE-SEED SUMMARY (mean +/- sample std) =====',flush=True)
    for mode in ('soft','hard'):
        ag=aggregate[mode]
        print(mode,'L2',f'{100*ag["relative_l2"]["mean"]:.5f} +/- {100*ag["relative_l2"]["std"]:.5f}%',
              'PDE',f'{ag["pde_rmse_heldout"]["mean"]:.5e} +/- {ag["pde_rmse_heldout"]["std"]:.5e}',
              flush=True)
    print('Paired errors (soft minus hard):',[round(x,7) for x in pairs['relative_l2']],flush=True)
    print('Saved outputs:', OUT, flush=True)


if __name__=='__main__':
    main()
