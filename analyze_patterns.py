import sys
sys.path.insert(0, '.')
from data_loader import load_data, get_active_contract, list_data_files
from ifvg_strategy import IFVGStrategy
import pandas as pd
import numpy as np

files = list_data_files()
raw_df = load_data(files)
df = get_active_contract(raw_df)

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
tdf = pd.DataFrame(trades)

tdf['entry_time'] = pd.to_datetime(tdf['entry_time'])
if tdf['entry_time'].dt.tz is not None:
    tdf['et_naive'] = tdf['entry_time'].dt.tz_localize(None)
else:
    tdf['et_naive'] = tdf['entry_time']
tdf['date'] = tdf['et_naive'].dt.date
tdf['is_win'] = tdf['result'] == 'WIN'

print("=== HYPOTHESIS 1: 2nd trade on same day has lower WR ===")
tdf['trade_num_today'] = tdf.groupby('date').cumcount() + 1
for tn in [1, 2]:
    subset = tdf[tdf['trade_num_today'] == tn]
    if len(subset) > 0:
        wr = subset['is_win'].mean() * 100
        pnl = subset['pnl_pts'].sum()
        print(f"  Trade #{tn}: {len(subset)} trades, WR={wr:.1f}%, P&L={pnl:+.1f} pts, avg_pnl={subset['pnl_pts'].mean():+.1f}")

print("\n=== HYPOTHESIS 2: High risk trades lose more ===")
for threshold in [10, 15, 18, 20]:
    high = tdf[tdf['risk_pts'] > threshold]
    low = tdf[tdf['risk_pts'] <= threshold]
    if len(high) > 0 and len(low) > 0:
        print(f"  Risk > {threshold}: {len(high)} trades, WR={high['is_win'].mean()*100:.1f}%, avg_pnl={high['pnl_pts'].mean():+.1f}")
        print(f"  Risk <= {threshold}: {len(low)} trades, WR={low['is_win'].mean()*100:.1f}%, avg_pnl={low['pnl_pts'].mean():+.1f}")
        print()

print("\n=== HYPOTHESIS 3: Entry time matters ===")
tdf['entry_hour_min'] = tdf['et_naive'].dt.hour * 60 + tdf['et_naive'].dt.minute
for start, end, label in [(585, 600, '9:45-10:00'), (600, 615, '10:00-10:15'), 
                            (615, 630, '10:15-10:30'), (630, 660, '10:30-11:00')]:
    subset = tdf[(tdf['entry_hour_min'] >= start) & (tdf['entry_hour_min'] < end)]
    if len(subset) >= 5:
        print(f"  {label}: {len(subset)} trades, WR={subset['is_win'].mean()*100:.1f}%, avg_pnl={subset['pnl_pts'].mean():+.1f}")

print("\n=== HYPOTHESIS 4: FVG size extremes ===")
for low, high in [(3, 5), (5, 10), (10, 20), (20, 50), (50, 200)]:
    subset = tdf[(tdf['fvg_size'] >= low) & (tdf['fvg_size'] < high)]
    if len(subset) >= 3:
        print(f"  FVG {low}-{high}: {len(subset)} trades, WR={subset['is_win'].mean()*100:.1f}%, avg_pnl={subset['pnl_pts'].mean():+.1f}")

print("\n=== HYPOTHESIS 5: Win rate after consecutive losses (same week/period) ===")
tdf_sorted = tdf.sort_values('entry_time').reset_index(drop=True)
prev_results = []
for i in range(len(tdf_sorted)):
    if i >= 2:
        last2 = tdf_sorted.iloc[i-2:i]['result'].values
        if all(r == 'LOSS' for r in last2):
            prev_results.append(('after_2L', tdf_sorted.iloc[i]))
    if i >= 1:
        last1 = tdf_sorted.iloc[i-1]['result']
        if last1 == 'LOSS':
            prev_results.append(('after_1L', tdf_sorted.iloc[i]))

for label in ['after_1L', 'after_2L']:
    subset = [t for l, t in prev_results if l == label]
    if subset:
        wins = sum(1 for t in subset if t['result'] == 'WIN')
        total = len(subset)
        print(f"  {label}: {total} trades, WR={wins/total*100:.1f}%")

print("\n=== HYPOTHESIS 6: RR achieved on wins (trailing stop cutting short?) ===")
wins = tdf[tdf['result'] == 'WIN']
for rr_low, rr_high in [(0, 0.5), (0.5, 0.8), (0.8, 1.0), (1.0, 1.2), (1.2, 2.0)]:
    subset = wins[(wins['rr_achieved'] >= rr_low) & (wins['rr_achieved'] < rr_high)]
    if len(subset) > 0:
        print(f"  RR {rr_low}-{rr_high}: {len(subset)} wins ({len(subset)/len(wins)*100:.0f}%), avg_pnl={subset['pnl_pts'].mean():+.1f}")

print("\n=== HYPOTHESIS 7: Day of week performance ===")
tdf['dow'] = tdf['et_naive'].dt.day_name()
for dow in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
    subset = tdf[tdf['dow'] == dow]
    if len(subset) > 0:
        print(f"  {dow}: {len(subset)} trades, WR={subset['is_win'].mean()*100:.1f}%, avg_pnl={subset['pnl_pts'].mean():+.1f}")

print("\n=== HYPOTHESIS 8: Direction bias by month ===")
tdf['year_month'] = tdf['et_naive'].dt.to_period('M').astype(str)
monthly_dir = tdf.groupby(['year_month', 'direction']).agg(
    trades=('result', 'count'),
    wins=('is_win', 'sum'),
    pnl=('pnl_pts', 'sum')
).reset_index()
monthly_dir['wr'] = (monthly_dir['wins'] / monthly_dir['trades'] * 100).round(1)

monthly_totals = tdf.groupby('year_month').agg(trades=('result', 'count'), wr=('is_win', 'mean')).reset_index()
monthly_totals['wr'] = (monthly_totals['wr'] * 100).round(1)
weak_months = monthly_totals[(monthly_totals['wr'] < 60) & (monthly_totals['trades'] >= 3)]['year_month'].values

print("Direction WR in weak months:")
for ym in weak_months:
    month_data = monthly_dir[monthly_dir['year_month'] == ym]
    buy_data = month_data[month_data['direction'] == 'BUY']
    sell_data = month_data[month_data['direction'] == 'SELL']
    buy_wr = buy_data['wr'].values[0] if len(buy_data) > 0 else 0
    sell_wr = sell_data['wr'].values[0] if len(sell_data) > 0 else 0
    buy_n = buy_data['trades'].values[0] if len(buy_data) > 0 else 0
    sell_n = sell_data['trades'].values[0] if len(sell_data) > 0 else 0
    print(f"  {ym}: BUY {buy_n}t/{buy_wr:.0f}% | SELL {sell_n}t/{sell_wr:.0f}%")

print("\n=== SIMULATION: What if max 1 trade/day? ===")
one_per_day = tdf[tdf['trade_num_today'] == 1]
ym_1pd = one_per_day.groupby('year_month').agg(
    trades=('result', 'count'),
    wins=('is_win', 'sum'),
    pnl=('pnl_pts', 'sum')
).reset_index()
ym_1pd['wr'] = (ym_1pd['wins'] / ym_1pd['trades'] * 100).round(1)
q_1pd = ym_1pd[ym_1pd['trades'] >= 3]
below_60_1pd = (q_1pd['wr'] < 60).sum()
print(f"  Total trades: {len(one_per_day)}, WR: {one_per_day['is_win'].mean()*100:.1f}%")
print(f"  Months below 60%: {below_60_1pd}/{len(q_1pd)}")

print("\n=== SIMULATION: What if max risk 15 pts? ===")
low_risk = tdf[tdf['risk_pts'] <= 15]
ym_lr = low_risk.groupby('year_month').agg(
    trades=('result', 'count'),
    wins=('is_win', 'sum'),
    pnl=('pnl_pts', 'sum')
).reset_index()
ym_lr['wr'] = (ym_lr['wins'] / ym_lr['trades'] * 100).round(1)
q_lr = ym_lr[ym_lr['trades'] >= 3]
below_60_lr = (q_lr['wr'] < 60).sum()
print(f"  Total trades: {len(low_risk)}, WR: {low_risk['is_win'].mean()*100:.1f}%")
print(f"  Months below 60%: {below_60_lr}/{len(q_lr)}")

print("\n=== COMBINED: 1 trade/day + max risk 15 ===")
combined = tdf[(tdf['trade_num_today'] == 1) & (tdf['risk_pts'] <= 15)]
ym_c = combined.groupby('year_month').agg(
    trades=('result', 'count'),
    wins=('is_win', 'sum'),
    pnl=('pnl_pts', 'sum')
).reset_index()
ym_c['wr'] = (ym_c['wins'] / ym_c['trades'] * 100).round(1)
q_c = ym_c[ym_c['trades'] >= 3]
below_60_c = (q_c['wr'] < 60).sum()
print(f"  Total trades: {len(combined)}, WR: {combined['is_win'].mean()*100:.1f}%")
print(f"  Months below 60%: {below_60_c}/{len(q_c)}")
print(f"  P&L: {combined['pnl_pts'].sum():+.1f}")
for _, row in ym_c.iterrows():
    status = ""
    if row['trades'] >= 3:
        if row['wr'] < 50: status = "BAD"
        elif row['wr'] < 60: status = "WEAK"
        else: status = "OK"
    print(f"    {row['year_month']}: {row['trades']}t, WR={row['wr']:.0f}%, P&L={row['pnl']:+.1f} {status}")
