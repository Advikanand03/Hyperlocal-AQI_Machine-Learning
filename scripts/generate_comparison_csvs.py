import pandas as pd
from pathlib import Path

root = Path(__file__).resolve().parent.parent
file_patterns = [
    'outputs/baseline_results.csv',
    'outputs/baseline_linear.csv',
    'outputs/baseline_xgboost.csv',
    'outputs/baseline_lightgbm.csv',
    'outputs/baseline_catboost.csv',
]
frames = []
for fp in file_patterns:
    path = root / fp
    if path.exists():
        print('loading', fp)
        frames.append(pd.read_csv(path))
    else:
        print('missing', fp)

if not frames:
    raise SystemExit('No files found to combine')

all_df = pd.concat(frames, ignore_index=True)
print('combined rows', len(all_df), 'cols', all_df.columns.tolist())

for test_name, out_file in [
    ('A_central', 'outputs/comparison_test_A.csv'),
    ('B_peripheral', 'outputs/comparison_test_B.csv'),
    ('C_LOSO', 'outputs/comparison_test_C.csv'),
]:
    sub = all_df[all_df['test'] == test_name]
    if sub.empty:
        print('warning: no data for', test_name)
        continue
    sub = sub[['model', 'station', 'MAE', 'RMSE', 'R2']]
    out_path = root / out_file
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_path, index=False)
    print('wrote', out_path, 'records', len(sub))
