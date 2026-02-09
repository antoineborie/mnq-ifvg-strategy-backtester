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
    print(f"Loaded {len(df)} rows ({len(raw)} raw) from {len(files)} files")
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

    trades_df['year'] = pd.to_datetime(trades_df['entry_time']).dt.year
    yearly = {int(y): round((g['result']=='WIN').sum()/len(g)*100,1) for y, g in trades_df.groupby('year')}

    return {
        'trades': m['total_trades'], 'wr': m['win_rate'], 'pnl': m['total_pnl_pts'],
        'pf': m['profit_factor'], 'dd': m['max_drawdown_pts'],
        'tpw': m['trades_per_week'], 'con': round(consistency,1),
        'nm': len(qual), 'floor': round(wr_floor,1), 'yr': yearly,
        'streak': m['max_consecutive_losses'],
    }

if __name__ == '__main__':
    df = load_all_data()
    
    fixed = {
        'use_trailing_stop': True, 'trail_trigger_rr': 0.5, 'trail_offset_pct': 30,
        'use_be': True, 'be_trigger_rr': 0.5, 'max_risk_pts': 25.0, 'min_risk_pts': 5.0,
        'cooldown_minutes': 10, 'use_displacement_filter': True, 'min_displacement_body_pct': 55,
        'min_displacement_size': 3.5, 'use_m1_confirmation': True, 'contract_value': 2.0,
        'killzone_start': '09:30', 'killzone_end': '11:00',
    }

    configs = [
        ("E1: BASELINE 1TPD/age8/10:05/SAL/OR", {'max_trades_per_day': 1, 'entry_start_time': '10:05', 'max_fvg_age_m15': 8, 'rr_target': 1.2, 'use_stop_after_loss': True, 'use_opening_range_filter': True, 'min_fvg_size': 3.0, 'retracement_pct': 60}),
        ("A1: 2TPD/age12/9:50/SAL/noOR", {'max_trades_per_day': 2, 'entry_start_time': '09:50', 'max_fvg_age_m15': 12, 'rr_target': 1.2, 'use_stop_after_loss': True, 'use_opening_range_filter': False, 'min_fvg_size': 3.0, 'retracement_pct': 60}),
        ("A2: 2TPD/age12/9:45/SAL/noOR", {'max_trades_per_day': 2, 'entry_start_time': '09:45', 'max_fvg_age_m15': 12, 'rr_target': 1.2, 'use_stop_after_loss': True, 'use_opening_range_filter': False, 'min_fvg_size': 3.0, 'retracement_pct': 60}),
        ("A3: 2TPD/age15/9:50/SAL/noOR", {'max_trades_per_day': 2, 'entry_start_time': '09:50', 'max_fvg_age_m15': 15, 'rr_target': 1.2, 'use_stop_after_loss': True, 'use_opening_range_filter': False, 'min_fvg_size': 3.0, 'retracement_pct': 60}),
        ("A4: 2TPD/age15/9:45/SAL/noOR", {'max_trades_per_day': 2, 'entry_start_time': '09:45', 'max_fvg_age_m15': 15, 'rr_target': 1.2, 'use_stop_after_loss': True, 'use_opening_range_filter': False, 'min_fvg_size': 3.0, 'retracement_pct': 60}),
        ("A5: 2TPD/age12/9:50/noSAL/noOR", {'max_trades_per_day': 2, 'entry_start_time': '09:50', 'max_fvg_age_m15': 12, 'rr_target': 1.2, 'use_stop_after_loss': False, 'use_opening_range_filter': False, 'min_fvg_size': 3.0, 'retracement_pct': 60}),
        ("A6: 2TPD/age12/9:45/noSAL/noOR", {'max_trades_per_day': 2, 'entry_start_time': '09:45', 'max_fvg_age_m15': 12, 'rr_target': 1.2, 'use_stop_after_loss': False, 'use_opening_range_filter': False, 'min_fvg_size': 3.0, 'retracement_pct': 60}),
        ("A7: 2TPD/age15/9:50/noSAL/noOR", {'max_trades_per_day': 2, 'entry_start_time': '09:50', 'max_fvg_age_m15': 15, 'rr_target': 1.2, 'use_stop_after_loss': False, 'use_opening_range_filter': False, 'min_fvg_size': 3.0, 'retracement_pct': 60}),
        ("B1: 2TPD/age12/9:50/SAL/noOR/rr1.0", {'max_trades_per_day': 2, 'entry_start_time': '09:50', 'max_fvg_age_m15': 12, 'rr_target': 1.0, 'use_stop_after_loss': True, 'use_opening_range_filter': False, 'min_fvg_size': 3.0, 'retracement_pct': 60}),
        ("B2: 2TPD/age12/9:50/SAL/noOR/fvg2.0", {'max_trades_per_day': 2, 'entry_start_time': '09:50', 'max_fvg_age_m15': 12, 'rr_target': 1.2, 'use_stop_after_loss': True, 'use_opening_range_filter': False, 'min_fvg_size': 2.0, 'retracement_pct': 60}),
        ("B3: 2TPD/age12/9:50/SAL/noOR/ret50", {'max_trades_per_day': 2, 'entry_start_time': '09:50', 'max_fvg_age_m15': 12, 'rr_target': 1.2, 'use_stop_after_loss': True, 'use_opening_range_filter': False, 'min_fvg_size': 3.0, 'retracement_pct': 50}),
        ("C1: 1TPD/age12/9:45/SAL/noOR", {'max_trades_per_day': 1, 'entry_start_time': '09:45', 'max_fvg_age_m15': 12, 'rr_target': 1.2, 'use_stop_after_loss': True, 'use_opening_range_filter': False, 'min_fvg_size': 3.0, 'retracement_pct': 60}),
        ("C2: 1TPD/age15/9:45/SAL/noOR", {'max_trades_per_day': 1, 'entry_start_time': '09:45', 'max_fvg_age_m15': 15, 'rr_target': 1.2, 'use_stop_after_loss': True, 'use_opening_range_filter': False, 'min_fvg_size': 3.0, 'retracement_pct': 60}),
        ("D1: 2TPD/age15/9:45/SAL/noOR/fvg2/ret50", {'max_trades_per_day': 2, 'entry_start_time': '09:45', 'max_fvg_age_m15': 15, 'rr_target': 1.2, 'use_stop_after_loss': True, 'use_opening_range_filter': False, 'min_fvg_size': 2.0, 'retracement_pct': 50}),
        ("D2: 2TPD/age15/9:45/noSAL/noOR/fvg2/ret50", {'max_trades_per_day': 2, 'entry_start_time': '09:45', 'max_fvg_age_m15': 15, 'rr_target': 1.2, 'use_stop_after_loss': False, 'use_opening_range_filter': False, 'min_fvg_size': 2.0, 'retracement_pct': 50}),
    ]

    print(f"Testing {len(configs)} configs...\n")
    for idx, (name, cfg) in enumerate(configs):
        t0 = time.time()
        full = {**fixed, **cfg}
        r = analyze(df, full)
        elapsed = time.time() - t0
        if r:
            print(f"{name}")
            print(f"  T:{r['trades']} WR:{r['wr']}% PnL:{r['pnl']} PF:{r['pf']} TPW:{r['tpw']} Con:{r['con']}% Floor:{r['floor']}% DD:{r['dd']} Strk:{r['streak']} ({elapsed:.1f}s)")
            print(f"  Yr: {r['yr']}")
        else:
            print(f"{name} - NO TRADES ({elapsed:.1f}s)")
        sys.stdout.flush()
