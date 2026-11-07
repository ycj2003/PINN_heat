"""Post-run reproducibility and consistency checks for the submitted result files."""
from pathlib import Path
import csv, json, math
import torch
ROOT=Path(__file__).resolve().parent

# Both benchmark exact solutions satisfy the stated PDE.
x=torch.rand(64,1)
t=torch.rand(64,1)
xt=torch.cat([x,t],1).requires_grad_(True)
x1,t1=xt[:,:1],xt[:,1:]
a=.1
u=torch.exp(-a*math.pi**2*t1)*torch.sin(math.pi*x1)
g=torch.autograd.grad(u,xt,torch.ones_like(u),create_graph=True)[0]
gxx=torch.autograd.grad(g[:,:1],xt,torch.ones_like(g[:,:1]),create_graph=True)[0][:,:1]
assert float((g[:,1:]-a*gxx).abs().max().detach())<1e-5

u2=torch.exp(-t1)*torch.sin(math.pi*x1)+.2*t1*torch.sin(3*math.pi*x1)
g2=torch.autograd.grad(u2,xt,torch.ones_like(u2),create_graph=True)[0]
gxx2=torch.autograd.grad(g2[:,:1],xt,torch.ones_like(g2[:,:1]),create_graph=True)[0][:,:1]
q=((a*math.pi**2-1)*torch.exp(-t1)*torch.sin(math.pi*x1)+.2*(1+9*a*math.pi**2*t1)*torch.sin(3*math.pi*x1))
assert float((g2[:,1:]-a*gxx2-q).abs().max().detach())<1e-5

# Hard conditions hold without optimization.
from paper_aligned import PINN, initial
m=PINN('hard')
z=torch.rand(41,1)
with torch.no_grad():
    assert torch.max(abs(m(torch.cat([z,torch.zeros_like(z)],1))-initial(z))).item()<1e-6
    assert torch.max(abs(m(torch.cat([torch.zeros_like(z),z],1)))).item()<1e-6
    assert torch.max(abs(m(torch.cat([torch.ones_like(z),z],1)))).item()<1e-6

# All five seeds completed exactly 2,000 epochs for both methods.
ms=ROOT/'results'/'multi_seed'
for seed in range(1,6):
    for mode in ('soft','hard'):
        p=ms/f'{mode}_seed{seed}_history.csv'
        with p.open(newline='') as f: rows=list(csv.DictReader(f))
        assert len(rows)==2000,(p,len(rows))
        assert int(rows[-1]['epoch'])==2000
        assert (ms/f'{mode}_seed{seed}_model.pt').exists()
        metrics=json.loads((ms/f'{mode}_seed{seed}_metrics.json').read_text())
        assert metrics['epochs']==2000
        assert 0<=metrics['relative_l2']<1

# New paper-aligned experiment exists with both completed stages.
paper=ROOT/'results'/'paper_aligned'
metrics=json.loads((paper/'metrics.json').read_text())
for mode in ('soft','hard'):
    with (paper/f'{mode}_history.csv').open(newline='') as f: rows=list(csv.DictReader(f))
    assert len([r for r in rows if r['stage']=='Adam'])==2000
    assert len([r for r in rows if r['stage']=='LBFGS'])>1
    assert (paper/f'{mode}_model.pt').exists()
    for key in ('relative_l2','pde_rmse_heldout','training_seconds','r2'):
        assert math.isfinite(metrics[mode][key]),(mode,key)
assert metrics['hard']['relative_l2']<metrics['soft']['relative_l2']
print('PASS: two exact PDE residuals, hard IC/BC, ten 2k-epoch histories, both paper-aligned runs, finite metrics')
