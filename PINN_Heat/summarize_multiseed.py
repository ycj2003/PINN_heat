"""Aggregate 5 paired runs from saved JSON/CSV files without retraining.

Usage: python summarize_multiseed.py
Produces a per-seed CSV, mean +/- sample std summaries, and two PNGs.
"""
import csv
import json
from pathlib import Path
import numpy as np
import run_multiseed as exp

rows=[]
histories={}
for seed in exp.SEEDS:
    for mode in ('soft','hard'):
        mf=exp.OUT/f'{mode}_seed{seed}_metrics.json'
        hf=exp.OUT/f'{mode}_seed{seed}_history.csv'
        if not mf.exists() or not hf.exists():
            raise FileNotFoundError(f'Missing full run: seed={seed}, mode={mode}')
        metrics=json.loads(mf.read_text(encoding='utf8'))
        with hf.open('r',newline='',encoding='utf8') as f:
            data=list(csv.DictReader(f))
        if len(data)!=exp.EPOCHS or metrics['epochs']!=exp.EPOCHS:
            raise ValueError(f'Incomplete run: seed={seed}, mode={mode}')
        rows.append(metrics)
        histories[(seed,mode)]=[[int(r['epoch']),float(r['total']),float(r['physics']),
                                float(r['ic']),float(r['bc'])] for r in data]

fieldnames=['seed','mode','epochs','relative_l2','ic_rmse','bc_rmse','pde_rmse_heldout','training_seconds','device']
with (exp.OUT/'per_seed_metrics.csv').open('w',newline='',encoding='utf8') as f:
    writer=csv.DictWriter(f,fieldnames=fieldnames)
    writer.writeheader();writer.writerows(rows)

def stats(v):
    a=np.array(v,dtype=float)
    return dict(mean=float(a.mean()),sd=float(a.std(ddof=1)),min=float(a.min()),max=float(a.max()))

metrics=['relative_l2','ic_rmse','bc_rmse','pde_rmse_heldout','training_seconds']
agg={mode:{m:stats([r[m] for r in rows if r['mode']==mode]) for m in metrics}
     for mode in ('soft','hard')}
paired={m:stats([next(r[m] for r in rows if r['mode']=='soft' and r['seed']==seed)-
                   next(r[m] for r in rows if r['mode']=='hard' and r['seed']==seed)
                   for seed in exp.SEEDS]) for m in metrics}
meta=dict(benchmark='1D manufactured transient heat equation; source calculated from reference solution',
          seeds=list(exp.SEEDS),epochs_per_run=exp.EPOCHS,
          architecture='2-64-64-64-1, Tanh', optimizer='Adam',lr=exp.LR,
          alpha=exp.ALPHA,n_f_per_epoch=exp.N_F,n_ic=exp.N_IC,n_bc=exp.N_BC,
          lambda_ic=exp.LAMBDA_IC,lambda_bc=exp.LAMBDA_BC,
          pde_sample_seed_formula='314159 + seed, same Soft and Hard per pair',
          soft_ic_bc_sample_seed_formula='271828 + seed',
          heldout_pde_seed=123456,evaluation_grid_size=exp.GRID_SIZE,
          std_type='sample std (ddof=1), five independent paired seeds',
          numerical_results={"soft":agg['soft'],"hard":agg['hard']},
          paired_soft_minus_hard=paired,device='cpu')
(exp.OUT/'summary.json').write_text(json.dumps(meta,indent=2),encoding='utf8')

exp.plot_paired_relative_errors(rows)
exp.plot_pde_histories(histories)

summary_rows=[]
for m in metrics:
    summary_rows.append(dict(metric=m,soft_mean=agg['soft'][m]['mean'],soft_sd=agg['soft'][m]['sd'],
                             hard_mean=agg['hard'][m]['mean'],hard_sd=agg['hard'][m]['sd'],
                             soft_minus_hard_mean=paired[m]['mean'],soft_minus_hard_sd=paired[m]['sd']))
with (exp.OUT/'summary_table.csv').open('w',newline='',encoding='utf8') as f:
    w=csv.DictWriter(f,fieldnames=list(summary_rows[0].keys()))
    w.writeheader();w.writerows(summary_rows)

with (exp.OUT/'report_numbers.md').open('w',encoding='utf8') as f:
    f.write('# Measured five-seed results (2000 epochs per run)\n\n')
    f.write('All results use the same 101x101 exact-solution evaluation grid and a held-out set of 2048 physics points. '\
            'Values below are mean ± sample standard deviation (n=5).\n\n')
    f.write('| Metric | Soft | Hard |\n|---|---:|---:|\n')
    for metric,scale,fmt in [('relative_l2',100,'.4f'),('pde_rmse_heldout',1,'.5f'),('ic_rmse',1,'.6f'),('bc_rmse',1,'.9f'),('training_seconds',1,'.2f')]:
        suffix=' (%)' if scale==100 else ''
        a,b=agg['soft'][metric],agg['hard'][metric]
        f.write(f'| {metric}{suffix} | {format(scale*a["mean"],fmt)} ± {format(scale*a["sd"],fmt)} | '
                f'{format(scale*b["mean"],fmt)} ± {format(scale*b["sd"],fmt)} |\n')
    f.write('\n## Per-seed relative L2 error (%)\n\n| Seed | Soft | Hard |\n|---:|---:|---:|\n')
    for seed in exp.SEEDS:
        a=100*next(r['relative_l2'] for r in rows if r['mode']=='soft' and r['seed']==seed)
        b=100*next(r['relative_l2'] for r in rows if r['mode']=='hard' and r['seed']==seed)
        f.write(f'| {seed} | {a:.4f} | {b:.4f} |\n')
    f.write('\nThese results are for a manufactured-solution benchmark, not a direct reproduction '
            'of the source paper data. Five seeds do not establish universal superiority. '
            'The hard ansatz enforces IC/BC only, not the PDE.\n')

print((exp.OUT/'report_numbers.md').read_text(encoding='utf8'))
print('Aggregated 10 full 2000-epoch runs. Outputs at:',exp.OUT)
