# Measured five-seed results (2000 epochs per run)

All results use the same 101x101 exact-solution evaluation grid and a held-out set of 2048 physics points. Values below are mean ± sample standard deviation (n=5).

| Metric | Soft | Hard |
|---|---:|---:|
| relative_l2 (%) | 1.1009 ± 0.6064 | 0.1331 ± 0.0559 |
| pde_rmse_heldout | 0.03582 ± 0.00686 | 0.00659 ± 0.00223 |
| ic_rmse | 0.004312 ± 0.002161 | 0.000000 ± 0.000000 |
| bc_rmse | 0.005573581 ± 0.002553092 | 0.000000062 ± 0.000000000 |
| training_seconds | 11.67 ± 1.76 | 10.70 ± 1.28 |

## Per-seed relative L2 error (%)

| Seed | Soft | Hard |
|---:|---:|---:|
| 1 | 1.9197 | 0.1791 |
| 2 | 0.6480 | 0.0902 |
| 3 | 0.8692 | 0.0597 |
| 4 | 1.5497 | 0.1864 |
| 5 | 0.5177 | 0.1501 |

These results are for a manufactured-solution benchmark, not a direct reproduction of the source paper data. Five seeds do not establish universal superiority. The hard ansatz enforces IC/BC only, not the PDE.
