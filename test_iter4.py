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

    max_ls = 0
    streak = 0
    for r in tdf['result'].values:
        if r == 'LOSS':
            streak += 1
            max_ls = max(max_ls, streak)
        else:
            streak = 0

    print(f"  {name}: {total_trades}t WR={overall_wr:.1f}% P&L={total_pnl:+.0f} "
          f"TPW={tpw:.1f} B60={below_60}/{total_q} CONS={consistency:.0f}% "
          f"B50={below_50} MLS={max_ls}")

    if verbose:
        weak_months = []
        for _, row in monthly.iterrows():
            flag = ""
            if row['trades'] >= 3:
                if row['wr'] < 50: flag = " <<BAD"
                elif row['wr'] < 60: flag = " <WEAK"
            if flag:
                weak_months.append(row['year_month'])
            print(f"    {row['year_month']}: {row['trades']}t WR={row['wr']:.0f}% P&L={row['pnl']:+.1f}{flag}")

    return {
        'name': name, 'trades': total_trades, 'wr': overall_wr, 'pnl': total_pnl,
        'below_60': below_60, 'total_q': total_q, 'consistency': consistency,
        'tpw': tpw, 'below_50': below_50, 'max_ls': max_ls,
    }

base = {
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
print("ITERATION 4: TARGETED COMBINATIONS")
print("=" * 80)

results = []

print("\n--- Base + OR filter combinations ---")
for est in ['10:00', '10:05', '10:10']:
    cfg = {**base, 'entry_start_time': est, 'use_opening_range_filter': True}
    r = test_config(f"OR+EST={est}", cfg)
    if r: results.append(r)

print("\n--- OR + displacement variations ---")
for body in [55, 60, 65]:
    cfg = {**base, 'use_opening_range_filter': True, 'min_displacement_body_pct': body}
    r = test_config(f"OR+BODY={body}", cfg)
    if r: results.append(r)

print("\n--- OR + FVG age ---")
for age in [8, 10, 12]:
    cfg = {**base, 'use_opening_range_filter': True, 'max_fvg_age_m15': age}
    r = test_config(f"OR+AGE={age}", cfg)
    if r: results.append(r)

print("\n--- OR + 1 trade/day ---")
r = test_config("OR+1TPD", {**base, 'use_opening_range_filter': True, 'max_trades_per_day': 1})
if r: results.append(r)

print("\n--- Best combos with OR filter ---")
for est in ['10:05', '10:10']:
    for body in [55, 60]:
        for age in [10, 12, 15]:
            for ret in [50, 60]:
                cfg = {**base, 'entry_start_time': est,
                       'use_opening_range_filter': True,
                       'min_displacement_body_pct': body,
                       'max_fvg_age_m15': age,
                       'retracement_pct': ret}
                r = test_config(f"OR+{est}/B{body}/A{age}/R{ret}", cfg)
                if r: results.append(r)

print("\n--- Enhanced bias + OR ---")
for conf in [2, 3]:
    cfg = {**base, 'use_opening_range_filter': True,
           'use_enhanced_bias': True, 'bias_min_confidence': conf}
    r = test_config(f"OR+BIAS{conf}", cfg)
    if r: results.append(r)

print("\n--- Multi-confirm + OR ---")
for br in [35, 40, 45]:
    cfg = {**base, 'use_opening_range_filter': True,
           'use_multi_confirm': True, 'use_m1_confirmation': False,
           'confirm_require_body_ratio': br}
    r = test_config(f"OR+ECONF={br}", cfg)
    if r: results.append(r)

print("\n--- Extended killzone ---")
for ke in ['11:00', '11:30', '12:00']:
    cfg = {**base, 'killzone_end': ke}
    r = test_config(f"KZ_END={ke}", cfg)
    if r: results.append(r)

print("\n--- Extended killzone + OR ---")
for ke in ['11:30', '12:00']:
    cfg = {**base, 'killzone_end': ke, 'use_opening_range_filter': True}
    r = test_config(f"KZ={ke}+OR", cfg)
    if r: results.append(r)

print("\n" + "=" * 80)
print("TOP 15 CONFIGS BY CONSISTENCY (then WR, then P&L):")
print("=" * 80)
results.sort(key=lambda x: (-x['consistency'], x['below_60'], -x['wr'], -x['pnl']))
for i, r in enumerate(results[:15]):
    print(f"{i+1}. {r['name']}: {r['trades']}t WR={r['wr']:.1f}% P&L={r['pnl']:+.0f} "
          f"TPW={r['tpw']:.1f} B60={r['below_60']}/{r['total_q']} CONS={r['consistency']:.0f}% B50={r['below_50']}")

if results:
    best = results[0]
    print(f"\n--- BEST CONFIG FULL DETAILS ---")
    test_config(f"BEST: {best['name']}", best.get('config', base), verbose=True)
