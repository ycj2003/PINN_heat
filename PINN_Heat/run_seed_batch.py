"""Resume-capable seed runner; e.g. python run_seed_batch.py 2 3.

Runs exactly 2000 epochs per missing seed/mode and saves metrics immediately.
"""
import sys, csv, json, re
from pathlib import Path
import torch
import run_multiseed as exp

seed_args = [int(v) for v in sys.argv[1:]] or list(exp.SEEDS)
for seed in seed_args:
    for mode in ('soft','hard'):
        metric_path = exp.OUT/f'{mode}_seed{seed}_metrics.json'
        hist_path = exp.OUT/f'{mode}_seed{seed}_history.csv'
        model_path = exp.OUT/f'{mode}_seed{seed}_model.pt'
        if metric_path.exists():
            print(f'EXIST seed={seed} mode={mode} -> skip',flush=True)
            continue
        if model_path.exists() and hist_path.exists():
            # Earlier process ended before summary, but this run was completed.
            torch.manual_seed(seed)
            model = exp.HeatPINN(mode)
            model.load_state_dict(torch.load(model_path,map_location='cpu',weights_only=True))
            log=(exp.ROOT/'results'/'multi_seed_train_log.txt').read_text(encoding='utf8')
            match=re.search(rf'RESULT seed={seed} {mode}:.*? elapsed=([0-9.]+)s',log)
            elapsed=float(match.group(1)) if match else float('nan')
            print(f'RECOVER seed={seed} mode={mode} saved full model and history',flush=True)
        else:
            model, history, elapsed = exp.train_one(seed,mode)
        m = exp.evaluate_one(seed,mode,model,elapsed)
        metric_path.write_text(json.dumps(m,indent=2),encoding='utf8')
        print(f'METRIC seed={seed} {mode} rel_L2={m["relative_l2"]:.9f} PDE_RMSE={m["pde_rmse_heldout"]:.9f} secs={m["training_seconds"]}',flush=True)
        del model
