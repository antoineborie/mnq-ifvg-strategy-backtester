import sys
sys.path.insert(0, '.')
from data_loader import load_data, get_active_contract, list_data_files
from ifvg_strategy import IFVGStrategy
import pandas as pd
import numpy as np

files = list_data_files()
raw_df = load_data(files)
df = get_active_contract(raw_df)

def test_config(name, config, verbose=False):
    strategy = IFVGStrategy(config)
    results = strategy.run_backtest(df)
    tdf = results['trades']
    if isinstance(tdf, list):
        tdf = pd.DataFrame(tdf)
    if tdf is None or len(tdf) == 0:
        return None

    tdf['entry_time'] = pd.to_datetime(tdf['entry_time'])
    if tdf['entry_time'].dt.tz is not None:
        tdf['et_naive'] = tdf['entry_time'].dt.tz_localize(None)
    else:
        tdf['et_naive'] = tdf['entry_time']
    tdf['year_month'] = tdf['et_naive'].dt.to_period('M').astype(str)
    tdf['is_win'] = tdf['result'] == 'WIN'

    monthly = tdf.groupby('year_month').agg(
        trades=('result', 'count'),
        wins=('is_win', 'sum'),
        pnl=('pnl_pts', 'sum')
    ).reset_index()
    monthly['wr'] = (monthly['wins'] / monthly['trades'] * 100).round(1)

    qualified = monthly[monthly['trades'] >= 3]
    below_60 = (qualified['wr'] < 60).sum()
    below_50 = (qualified['wr'] < 50).sum()
    total_q = len(qualified)
    consistency = (total_q - below_60) / total_q * 100 if total_q > 0 else 0

    total_trades = len(tdf)
    overall_wr = tdf['is_win'].mean() * 100
    total_pnl = tdf['pnl_pts'].sum()

    first_date = tdf['et_naive'].min()
    last_date = tdf['et_naive'].max()
    weeks = max(1, (last_date - first_date).days / 7)
    tpw = total_trades / weeks

    max_loss_streak = 0
    streak = 0
    for r in tdf['result'].values:
        if r == 'LOSS':
            streak += 1
            max_loss_streak = max(max_loss_streak, streak)
        else:
            streak = 0

    print(f"  {name}: {total_trades}t WR={overall_wr:.1f}% P&L={total_pnl:+.0f} "
          f"TPW={tpw:.1f} B60={below_60}/{total_q} CONS={consistency:.0f}% "
          f"B50={below_50} MLS={max_loss_streak}")

    if verbose:
        for _, row in monthly.iterrows():
            flag = ""
            if row['trades'] >= 3:
                if row['wr'] < 50: flag = " <<BAD"
                elif row['wr'] < 60: flag = " <WEAK"
            print(f"    {row['year_month']}: {row['trades']}t WR={row['wr']:.0f}% P&L={row['pnl']:+.1f}{flag}")

    return {
        'name': name, 'trades': total_trades, 'wr': overall_wr, 'pnl': total_pnl,
        'below_60': below_60, 'total_q': total_q, 'consistency': consistency,
        'tpw': tpw, 'below_50': below_50, 'max_loss_streak': max_loss_streak,
        'config': config,
    }

best1005 = {
    'min_fvg_size': 3.0, 'max_fvg_age_m15': 15, 'rr_target': 1.2,
    'max_risk_pts': 25.0, 'min_risk_pts': 5.0, 'max_trades_per_day': 2,
    'killzone_start': '09:30', 'killzone_end': '11:00',
    'use_be': True, 'be_trigger_rr': 0.5, 'cooldown_minutes': 10,
    'contract_value': 2.0, 'target_mode': 'fixed_rr', 'retracement_pct': 60,
    'structure_lookback_days': 20, 'use_displacement_filter': True,
    'min_displacement_body_pct': 55, 'min_displacement_size': 3.5,
    'use_m1_confirmation': True, 'use_trailing_stop': True,
    'trail_trigger_rr': 0.5, 'trail_offset_pct': 30,
    'entry_start_time': '10:05',
    'use_stop_after_loss': True,
}

print("=" * 80)
print("ITERATION 3: BUILDING ON EST=10:05 + STOP AFTER LOSS (B60=8)")
print("=" * 80)

results = []

print("\n--- Baseline 10:05 ---")
r = test_config("BASE_1005", best1005, verbose=True)
if r: results.append(r)

print("\n--- Add enhanced bias ---")
for conf in [2, 3]:
    cfg = {**best1005, 'use_enhanced_bias': True, 'bias_min_confidence': conf}
    r = test_config(f"BIAS_CONF={conf}", cfg)
    if r: results.append(r)

print("\n--- Add opening range filter ---")
r = test_config("OR_FILTER", {**best1005, 'use_opening_range_filter': True})
if r: results.append(r)

print("\n--- Vary BE trigger with 10:05 base ---")
for be in [0.4, 0.5, 0.6, 0.7, 0.8]:
    r = test_config(f"BE={be}", {**best1005, 'be_trigger_rr': be})
    if r: results.append(r)

print("\n--- Vary trail with 10:05 base ---")
for tt, to in [(0.5, 25), (0.5, 30), (0.5, 35), (0.6, 30), (0.6, 35), (0.6, 40), (0.7, 35), (0.7, 40)]:
    r = test_config(f"TRAIL={tt}/{to}", {**best1005, 'trail_trigger_rr': tt, 'trail_offset_pct': to})
    if r: results.append(r)

print("\n--- Vary max FVG age ---")
for age in [8, 10, 12, 15]:
    r = test_config(f"AGE={age}", {**best1005, 'max_fvg_age_m15': age})
    if r: results.append(r)

print("\n--- Vary retracement ---")
for ret in [40, 50, 55, 60, 70]:
    r = test_config(f"RET={ret}", {**best1005, 'retracement_pct': ret})
    if r: results.append(r)

print("\n--- Vary displacement body ---")
for body in [45, 50, 55, 60, 65, 70]:
    r = test_config(f"BODY={body}", {**best1005, 'min_displacement_body_pct': body})
    if r: results.append(r)

print("\n--- Vary FVG size ---")
for fvg in [3.0, 3.5, 4.0, 4.5, 5.0]:
    r = test_config(f"FVG={fvg}", {**best1005, 'min_fvg_size': fvg})
    if r: results.append(r)

print("\n--- Vary max risk ---")
for mr in [15, 18, 20, 25]:
    r = test_config(f"RISK={mr}", {**best1005, 'max_risk_pts': mr})
    if r: results.append(r)

print("\n--- Enhanced confirm ---")
for br in [35, 40, 45, 50]:
    cfg = {**best1005, 'use_multi_confirm': True, 'use_m1_confirmation': False,
           'confirm_require_body_ratio': br}
    r = test_config(f"ECONF_BR={br}", cfg)
    if r: results.append(r)

print("\n--- 1 trade/day ---")
r = test_config("1TPD", {**best1005, 'max_trades_per_day': 1})
if r: results.append(r)

print("\n" + "=" * 80)
print("TOP 10 CONFIGS BY CONSISTENCY:")
print("=" * 80)
results.sort(key=lambda x: (-x['consistency'], -x['wr'], -x['pnl']))
for i, r in enumerate(results[:10]):
    print(f"{i+1}. {r['name']}: {r['trades']}t WR={r['wr']:.1f}% P&L={r['pnl']:+.0f} "
          f"TPW={r['tpw']:.1f} B60={r['below_60']}/{r['total_q']} CONS={r['consistency']:.0f}%")

best = results[0]
print(f"\n--- BEST CONFIG DETAILS ---")
test_config("BEST", best['config'], verbose=True)
