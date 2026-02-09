import os
import pandas as pd
import numpy as np
from data_loader import load_data, list_data_files, get_active_contract
from ifvg_strategy import IFVGStrategy

data_files = list_data_files()
files_2024_plus = [f for f in data_files if any(y in os.path.basename(f) for y in ['2024', '2025', '2026'])]
raw_df = load_data(files_2024_plus)
df = get_active_contract(raw_df)

base = {
    'killzone_start': '09:30', 'killzone_end': '12:00', 'contract_value': 2.0,
    'target_mode': 'fixed_rr', 'structure_lookback_days': 20,
    'use_opening_range_filter': False, 'partial_tp_pct': 100,
    'use_liquidity_sweep': False, 'use_structure_confluence': False,
    'use_session_momentum': False, 'use_trend_filter': False,
    'use_range_filter': False, 'use_m1_momentum': False, 'use_day_filter': False,
    'min_risk_pts': 5.0, 'max_risk_pts': 25.0,
    'cooldown_minutes': 5, 'use_m1_confirmation': True, 'entry_start_time': '09:45',
    'min_displacement_body_pct': 55, 'min_displacement_size': 3.5,
    'use_be': False, 'use_trailing_stop': False, 'use_stop_after_loss': False,
    'retracement_pct': 50, 'use_displacement_filter': False,
}

def run_test(label, overrides):
    full = {**base, **overrides}
    strat = IFVGStrategy(full)
    strat.run_backtest(df)
    trades = strat.trades
    if len(trades) < 10:
        print(f"  {label:65s} {len(trades):3d}T")
        return None
    wins = sum(1 for t in trades if t['result'] == 'WIN')
    losses = sum(1 for t in trades if t['result'] == 'LOSS')
    bes = sum(1 for t in trades if t['result'] == 'BE')
    decisive = wins + losses
    wr = (wins / decisive * 100) if decisive > 0 else 0
    pnl = sum(t['pnl_pts'] for t in trades)
    dates = sorted(set(t['entry_time'].date() for t in trades))
    tpw = len(trades) / max(1, (dates[-1] - dates[0]).days / 7)
    monthly_pnl = {}
    for t in trades:
        ym = t['entry_time'].strftime('%Y-%m')
        monthly_pnl[ym] = monthly_pnl.get(ym, 0) + t['pnl_pts']
    months_count = len(monthly_pnl)
    profitable_months = sum(1 for v in monthly_pnl.values() if v > 0)
    avg_monthly = pnl / months_count if months_count > 0 else 0
    month_pct = (profitable_months / months_count * 100) if months_count > 0 else 0
    pf_num = sum(t['pnl_pts'] for t in trades if t['pnl_pts'] > 0)
    pf_den = abs(sum(t['pnl_pts'] for t in trades if t['pnl_pts'] < 0))
    pf = pf_num / pf_den if pf_den > 0 else 999
    print(f"  {label:65s} {len(trades):3d}T {wins:3d}W/{losses:3d}L/{bes:3d}BE WR={wr:5.1f}% "
          f"PnL={pnl:6.0f} PF={pf:.2f} TPW={tpw:.1f} Mo+={month_pct:.0f}% avg={avg_monthly:.0f}")
    return {'label': label, 'wr': wr, 'pnl': pnl, 'tpw': tpw, 'month_pct': month_pct, 'avg_monthly': avg_monthly, 'monthly_pnl': monthly_pnl, 'trades': len(trades)}

print("=== BEST FOUND: RR=2.0 pure TP/SL with monthly breakdown ===")
r = run_test("BEST: RR=2 TPD=2 pure retr=50 M1 fvg=3 age=15",
    {'rr_target': 2.0, 'max_trades_per_day': 2, 'min_fvg_size': 3.0, 'max_fvg_age_m15': 15})
if r:
    print("\n  Monthly breakdown:")
    for ym in sorted(r['monthly_pnl'].keys()):
        print(f"    {ym}: {r['monthly_pnl'][ym]:+7.1f}pts {'OK' if r['monthly_pnl'][ym] > 0 else 'XX'}")

print("\n=== FVG size variations ===")
for fvg in [1.0, 2.0, 3.0, 4.0, 5.0]:
    run_test(f"RR=2 fvg={fvg} age=15",
             {'rr_target': 2.0, 'max_trades_per_day': 2, 'min_fvg_size': fvg, 'max_fvg_age_m15': 15})

print("\n=== FVG age variations ===")
for age in [8, 10, 15, 20, 30]:
    run_test(f"RR=2 fvg=3 age={age}",
             {'rr_target': 2.0, 'max_trades_per_day': 2, 'min_fvg_size': 3.0, 'max_fvg_age_m15': age})

print("\n=== Risk bounds variations ===")
for maxr in [15, 20, 25, 30, 40]:
    run_test(f"RR=2 maxR={maxr}",
             {'rr_target': 2.0, 'max_trades_per_day': 2, 'min_fvg_size': 3.0, 'max_fvg_age_m15': 15, 'max_risk_pts': maxr})

print("\n=== Killzone end variations ===")
for kz_end in ['11:00', '11:30', '12:00', '12:30', '13:00']:
    run_test(f"RR=2 kzEnd={kz_end}",
             {'rr_target': 2.0, 'max_trades_per_day': 2, 'min_fvg_size': 3.0, 'max_fvg_age_m15': 15, 'killzone_end': kz_end})

print("\n=== Combined best with trailing for comparison ===")
r2 = run_test("BEST+trail: BE@1.0 trail@1.5/40",
    {'rr_target': 2.0, 'max_trades_per_day': 2, 'min_fvg_size': 3.0, 'max_fvg_age_m15': 15,
     'use_be': True, 'be_trigger_rr': 1.0, 'use_trailing_stop': True, 'trail_trigger_rr': 1.5, 'trail_offset_pct': 40})
if r2:
    print("  Monthly:")
    for ym in sorted(r2['monthly_pnl'].keys()):
        print(f"    {ym}: {r2['monthly_pnl'][ym]:+7.1f}pts {'OK' if r2['monthly_pnl'][ym] > 0 else 'XX'}")

r3 = run_test("BEST+trail: BE@1.0 trail@1.8/50",
    {'rr_target': 2.0, 'max_trades_per_day': 2, 'min_fvg_size': 3.0, 'max_fvg_age_m15': 15,
     'use_be': True, 'be_trigger_rr': 1.0, 'use_trailing_stop': True, 'trail_trigger_rr': 1.8, 'trail_offset_pct': 50})
if r3:
    print("  Monthly:")
    for ym in sorted(r3['monthly_pnl'].keys()):
        print(f"    {ym}: {r3['monthly_pnl'][ym]:+7.1f}pts {'OK' if r3['monthly_pnl'][ym] > 0 else 'XX'}")
