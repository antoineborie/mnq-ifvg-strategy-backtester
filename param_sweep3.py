import pandas as pd
import numpy as np
import sys
import time
from data_loader import load_data, list_data_files, get_active_contract
from ifvg_strategy import IFVGStrategy

def load_all_data():
    files = list_data_files()
    raw = load_data(files)
    df = get_active_contract(raw)
    print(f"Loaded {len(df)} rows from {len(files)} files")
    return df

def analyze(df, config):
    strat = IFVGStrategy(config)
    results = strat.run_backtest(df)
    trades_df = results['trades']
    m = results['metrics']
    if len(trades_df) == 0:
        return None
    trades_df = trades_df.copy()
    trades_df['month'] = pd.to_datetime(trades_df['entry_time']).dt.to_period('M')
    monthly = trades_df.groupby('month').apply(
        lambda g: pd.Series({'trades': len(g), 'wr': (g['result']=='WIN').sum()/len(g)*100})
    )
    qual = monthly[monthly['trades'] >= 3]
    consistency = (qual['wr'] >= 60).sum() / len(qual) * 100 if len(qual) > 0 else 0
    wr_floor = qual['wr'].min() if len(qual) > 0 else 0
    months_below_50 = (qual['wr'] < 50).sum() if len(qual) > 0 else 0

    trades_df['year'] = pd.to_datetime(trades_df['entry_time']).dt.year
    yearly = {int(y): round((g['result']=='WIN').sum()/len(g)*100,1) for y, g in trades_df.groupby('year')}

    return {
        'trades': m['total_trades'], 'wr': m['win_rate'], 'pnl': m['total_pnl_pts'],
        'pf': m['profit_factor'], 'dd': m['max_drawdown_pts'],
        'tpw': m['trades_per_week'], 'con': round(consistency,1),
        'nm': len(qual), 'floor': round(wr_floor,1), 'yr': yearly,
        'streak': m['max_consecutive_losses'], 'below50': months_below_50,
    }

if __name__ == '__main__':
    df = load_all_data()
    
    base = {
        'use_be': True, 'be_trigger_rr': 0.5, 'max_risk_pts': 25.0, 'min_risk_pts': 5.0,
        'use_displacement_filter': True, 'use_m1_confirmation': True, 'contract_value': 2.0,
        'killzone_start': '09:30',
    }

    configs = [
        ("P1: KZ12/trail0.3/35/age12/SAL", {**base, 'max_trades_per_day': 2, 'entry_start_time': '09:45', 'max_fvg_age_m15': 12, 'rr_target': 1.2, 'use_stop_after_loss': True, 'use_opening_range_filter': False, 'min_fvg_size': 3.0, 'retracement_pct': 60, 'killzone_end': '12:00', 'use_trailing_stop': True, 'trail_trigger_rr': 0.3, 'trail_offset_pct': 35, 'cooldown_minutes': 10, 'min_displacement_body_pct': 55, 'min_displacement_size': 3.5}),
        ("P2: KZ12/trail0.3/35/age15/SAL", {**base, 'max_trades_per_day': 2, 'entry_start_time': '09:45', 'max_fvg_age_m15': 15, 'rr_target': 1.2, 'use_stop_after_loss': True, 'use_opening_range_filter': False, 'min_fvg_size': 3.0, 'retracement_pct': 60, 'killzone_end': '12:00', 'use_trailing_stop': True, 'trail_trigger_rr': 0.3, 'trail_offset_pct': 35, 'cooldown_minutes': 10, 'min_displacement_body_pct': 55, 'min_displacement_size': 3.5}),
        ("P3: KZ11:30/trail0.3/35/age12/SAL", {**base, 'max_trades_per_day': 2, 'entry_start_time': '09:45', 'max_fvg_age_m15': 12, 'rr_target': 1.2, 'use_stop_after_loss': True, 'use_opening_range_filter': False, 'min_fvg_size': 3.0, 'retracement_pct': 60, 'killzone_end': '11:30', 'use_trailing_stop': True, 'trail_trigger_rr': 0.3, 'trail_offset_pct': 35, 'cooldown_minutes': 10, 'min_displacement_body_pct': 55, 'min_displacement_size': 3.5}),
        ("P4: KZ12/trail0.3/30/age12/SAL", {**base, 'max_trades_per_day': 2, 'entry_start_time': '09:45', 'max_fvg_age_m15': 12, 'rr_target': 1.2, 'use_stop_after_loss': True, 'use_opening_range_filter': False, 'min_fvg_size': 3.0, 'retracement_pct': 60, 'killzone_end': '12:00', 'use_trailing_stop': True, 'trail_trigger_rr': 0.3, 'trail_offset_pct': 30, 'cooldown_minutes': 10, 'min_displacement_body_pct': 55, 'min_displacement_size': 3.5}),
        ("P5: KZ12/trail0.3/35/age12/SAL/OR", {**base, 'max_trades_per_day': 2, 'entry_start_time': '09:45', 'max_fvg_age_m15': 12, 'rr_target': 1.2, 'use_stop_after_loss': True, 'use_opening_range_filter': True, 'min_fvg_size': 3.0, 'retracement_pct': 60, 'killzone_end': '12:00', 'use_trailing_stop': True, 'trail_trigger_rr': 0.3, 'trail_offset_pct': 35, 'cooldown_minutes': 10, 'min_displacement_body_pct': 55, 'min_displacement_size': 3.5}),
        ("P6: KZ12/trail0.3/35/age15/SAL/OR", {**base, 'max_trades_per_day': 2, 'entry_start_time': '09:45', 'max_fvg_age_m15': 15, 'rr_target': 1.2, 'use_stop_after_loss': True, 'use_opening_range_filter': True, 'min_fvg_size': 3.0, 'retracement_pct': 60, 'killzone_end': '12:00', 'use_trailing_stop': True, 'trail_trigger_rr': 0.3, 'trail_offset_pct': 35, 'cooldown_minutes': 10, 'min_displacement_body_pct': 55, 'min_displacement_size': 3.5}),
        ("P7: KZ12/trail0.4/30/age12/SAL", {**base, 'max_trades_per_day': 2, 'entry_start_time': '09:45', 'max_fvg_age_m15': 12, 'rr_target': 1.2, 'use_stop_after_loss': True, 'use_opening_range_filter': False, 'min_fvg_size': 3.0, 'retracement_pct': 60, 'killzone_end': '12:00', 'use_trailing_stop': True, 'trail_trigger_rr': 0.4, 'trail_offset_pct': 30, 'cooldown_minutes': 10, 'min_displacement_body_pct': 55, 'min_displacement_size': 3.5}),
        ("P8: KZ12/trail0.3/35/age12/noSAL", {**base, 'max_trades_per_day': 2, 'entry_start_time': '09:45', 'max_fvg_age_m15': 12, 'rr_target': 1.2, 'use_stop_after_loss': False, 'use_opening_range_filter': False, 'min_fvg_size': 3.0, 'retracement_pct': 60, 'killzone_end': '12:00', 'use_trailing_stop': True, 'trail_trigger_rr': 0.3, 'trail_offset_pct': 35, 'cooldown_minutes': 10, 'min_displacement_body_pct': 55, 'min_displacement_size': 3.5}),
        ("P9: KZ12/trail0.3/35/age12/SAL/rr1.0", {**base, 'max_trades_per_day': 2, 'entry_start_time': '09:45', 'max_fvg_age_m15': 12, 'rr_target': 1.0, 'use_stop_after_loss': True, 'use_opening_range_filter': False, 'min_fvg_size': 3.0, 'retracement_pct': 60, 'killzone_end': '12:00', 'use_trailing_stop': True, 'trail_trigger_rr': 0.3, 'trail_offset_pct': 35, 'cooldown_minutes': 10, 'min_displacement_body_pct': 55, 'min_displacement_size': 3.5}),
        ("P10: KZ12/trail0.3/35/age12/SAL/disp65", {**base, 'max_trades_per_day': 2, 'entry_start_time': '09:45', 'max_fvg_age_m15': 12, 'rr_target': 1.2, 'use_stop_after_loss': True, 'use_opening_range_filter': False, 'min_fvg_size': 3.0, 'retracement_pct': 60, 'killzone_end': '12:00', 'use_trailing_stop': True, 'trail_trigger_rr': 0.3, 'trail_offset_pct': 35, 'cooldown_minutes': 10, 'min_displacement_body_pct': 65, 'min_displacement_size': 3.5}),
        ("P11: KZ11:30/trail0.3/35/age15/SAL", {**base, 'max_trades_per_day': 2, 'entry_start_time': '09:45', 'max_fvg_age_m15': 15, 'rr_target': 1.2, 'use_stop_after_loss': True, 'use_opening_range_filter': False, 'min_fvg_size': 3.0, 'retracement_pct': 60, 'killzone_end': '11:30', 'use_trailing_stop': True, 'trail_trigger_rr': 0.3, 'trail_offset_pct': 35, 'cooldown_minutes': 10, 'min_displacement_body_pct': 55, 'min_displacement_size': 3.5}),
        ("P12: KZ12/trail0.3/35/age12/SAL/9:50", {**base, 'max_trades_per_day': 2, 'entry_start_time': '09:50', 'max_fvg_age_m15': 12, 'rr_target': 1.2, 'use_stop_after_loss': True, 'use_opening_range_filter': False, 'min_fvg_size': 3.0, 'retracement_pct': 60, 'killzone_end': '12:00', 'use_trailing_stop': True, 'trail_trigger_rr': 0.3, 'trail_offset_pct': 35, 'cooldown_minutes': 10, 'min_displacement_body_pct': 55, 'min_displacement_size': 3.5}),
    ]

    print(f"Phase 3: Testing {len(configs)} configs...\n")
    for idx, (name, cfg) in enumerate(configs):
        t0 = time.time()
        r = analyze(df, cfg)
        elapsed = time.time() - t0
        if r:
            print(f"{name}")
            print(f"  T:{r['trades']} WR:{r['wr']}% PnL:{r['pnl']} PF:{r['pf']} TPW:{r['tpw']} Con:{r['con']}% Floor:{r['floor']}% DD:{r['dd']} Strk:{r['streak']} <50:{r['below50']} ({elapsed:.1f}s)")
            print(f"  Yr: {r['yr']}")
        else:
            print(f"{name} - NO TRADES ({elapsed:.1f}s)")
        sys.stdout.flush()
