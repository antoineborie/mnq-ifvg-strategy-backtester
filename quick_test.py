import pandas as pd
from data_loader import load_data, list_data_files
from ifvg_strategy import IFVGStrategy

files = list_data_files()
print(f"Files: {len(files)}")
df = load_data(files)
print(f"Rows: {len(df)}, Columns: {list(df.columns)}")
print(f"Index type: {type(df.index)}, tz: {df.index.tz}")
print(f"First: {df.index[0]}, Last: {df.index[-1]}")
print(f"Sample:\n{df.head()}")

strat = IFVGStrategy()
print(f"\nRunning baseline config...")
results = strat.run_backtest(df)
print(f"Trades: {len(results['trades'])}")
if len(results['trades']) > 0:
    print(f"Metrics: {results['metrics']}")
