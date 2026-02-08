import sys
sys.path.insert(0, '.')
from data_loader import load_data, get_active_contract, list_data_files
from ifvg_strategy import IFVGStrategy
import pandas as pd
import numpy as np

files = list_data_files()
print(f"Loading {len(files)} data files...")
raw_df = load_data(files)
df = get_active_contract(raw_df)
print(f"Data: {len(df)} rows, {df.index.min()} to {df.index.max()}")

config = {
    'min_fvg_size': 3.0,
    'max_fvg_age_m15': 15,
    'rr_target': 1.2,
    'max_risk_pts': 25.0,
    'min_risk_pts': 5.0,
    'max_trades_per_day': 2,
    'killzone_start': '09:30',
    'killzone_end': '11:00',
    'use_be': True,
    'be_trigger_rr': 0.5,
    'cooldown_minutes': 10,
    'contract_value': 2.0,
    'target_mode': 'fixed_rr',
    'retracement_pct': 60,
    'structure_lookback_days': 20,
    'use_displacement_filter': True,
    'min_displacement_body_pct': 55,
    'min_displacement_size': 3.5,
    'use_m1_confirmation': True,
    'use_trailing_stop': True,
    'trail_trigger_rr': 0.5,
    'trail_offset_pct': 30,
    'entry_start_time': '09:45',
}

strategy = IFVGStrategy(config)
results = strategy.run_backtest(df)
trades = results['trades']
trades_df = pd.DataFrame(trades)

print(f"\n=== BASELINE RESULTS ===")
print(f"Total trades: {len(trades_df)}")
wins = (trades_df['result'] == 'WIN').sum()
print(f"Win rate: {wins/len(trades_df)*100:.1f}%")
print(f"Total P&L: {trades_df['pnl_pts'].sum():.1f} pts")

trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
if trades_df['entry_time'].dt.tz is not None:
    trades_df['entry_time_naive'] = trades_df['entry_time'].dt.tz_localize(None)
else:
    trades_df['entry_time_naive'] = trades_df['entry_time']
trades_df['year_month'] = trades_df['entry_time_naive'].dt.to_period('M').astype(str)

print(f"\n=== MONTHLY BREAKDOWN ===")
print(f"{'Month':<10} {'Trades':>7} {'Wins':>5} {'WR%':>6} {'P&L':>8} {'Status':>10}")
print("-" * 55)

monthly = trades_df.groupby('year_month').agg(
    trades=('result', 'count'),
    wins=('result', lambda x: (x == 'WIN').sum()),
    pnl=('pnl_pts', 'sum')
).reset_index()
monthly['wr'] = (monthly['wins'] / monthly['trades'] * 100).round(1)

months_below_60 = 0
months_below_50 = 0
qualified_months = 0

for _, row in monthly.iterrows():
    status = ""
    if row['trades'] >= 3:
        qualified_months += 1
        if row['wr'] < 50:
            status = "** BAD **"
            months_below_50 += 1
            months_below_60 += 1
        elif row['wr'] < 60:
            status = "* WEAK *"
            months_below_60 += 1
        else:
            status = "OK"
    else:
        status = "(few)"
    print(f"{row['year_month']:<10} {row['trades']:>7} {row['wins']:>5} {row['wr']:>5.1f}% {row['pnl']:>+7.1f} {status:>10}")

print(f"\nQualified months (3+ trades): {qualified_months}")
print(f"Months below 60%: {months_below_60}")
print(f"Months below 50%: {months_below_50}")
if qualified_months > 0:
    print(f"Consistency: {(qualified_months - months_below_60)/qualified_months*100:.1f}% of months at 60%+")

below_60 = monthly[(monthly['wr'] < 60) & (monthly['trades'] >= 3)]
if len(below_60) > 0:
    print(f"\n=== WEAK MONTHS DETAIL ===")
    for _, row in below_60.iterrows():
        ym = row['year_month']
        month_trades = trades_df[trades_df['year_month'] == ym]
        print(f"\n--- {ym} ({row['wr']:.1f}% WR, {row['trades']} trades, {row['pnl']:+.1f} pts) ---")
        for _, t in month_trades.iterrows():
            print(f"  {t['entry_time_naive'].strftime('%Y-%m-%d %H:%M')} {t['direction']:>4} "
                  f"risk={t['risk_pts']:.1f} pnl={t['pnl_pts']:+.1f} {t['result']:>4} "
                  f"rr={t.get('rr_achieved', 0):.2f} fvg={t['fvg_size']:.1f}"
                  f" vol={t.get('vol_regime', 'n/a')}")

print(f"\n=== TRADE DISTRIBUTION ===")
directions = trades_df['direction'].value_counts()
print(f"BUY trades: {directions.get('BUY', 0)}")
print(f"SELL trades: {directions.get('SELL', 0)}")

results_dist = trades_df['result'].value_counts()
print(f"\nResult distribution:")
for r, c in results_dist.items():
    print(f"  {r}: {c} ({c/len(trades_df)*100:.1f}%)")

avg_win = trades_df[trades_df['result'] == 'WIN']['pnl_pts'].mean()
avg_loss = trades_df[trades_df['result'] == 'LOSS']['pnl_pts'].mean()
print(f"\nAvg WIN: {avg_win:+.1f} pts")
print(f"Avg LOSS: {avg_loss:+.1f} pts")
print(f"Avg risk: {trades_df['risk_pts'].mean():.1f} pts")
