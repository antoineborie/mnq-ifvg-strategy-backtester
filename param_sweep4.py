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
    
    p4_base = {
        'max_trades_per_day': 2, 'entry_start_time': '09:45', 'max_fvg_age_m15': 12,
        'rr_target': 1.2, 'use_stop_after_loss': True, 'use_opening_range_filter': False,
        'min_fvg_size': 3.0, 'retracement_pct': 60, 'killzone_end': '12:00',
        'use_trailing_stop': True, 'trail_trigger_rr': 0.3, 'trail_offset_pct': 30,
        'cooldown_minutes': 10, 'min_displacement_body_pct': 55, 'min_displacement_size': 3.5,
        'use_be': True, 'be_trigger_rr': 0.5, 'max_risk_pts': 25.0, 'min_risk_pts': 5.0,
        'use_displacement_filter': True, 'use_m1_confirmation': True, 'contract_value': 2.0,
        'killzone_start': '09:30',
    }

    configs = [
        ("P4-REF: base", dict(p4_base)),
        ("P4a: age15", {**p4_base, 'max_fvg_age_m15': 15}),
        ("P4b: age10", {**p4_base, 'max_fvg_age_m15': 10}),
        ("P4c: 9:50", {**p4_base, 'entry_start_time': '09:50'}),
        ("P4d: 10:00", {**p4_base, 'entry_start_time': '10:00'}),
        ("P4e: disp60", {**p4_base, 'min_displacement_body_pct': 60}),
        ("P4f: trail25", {**p4_base, 'trail_offset_pct': 25}),
        ("P4g: trail35", {**p4_base, 'trail_offset_pct': 35}),
        ("P4h: rr1.0", {**p4_base, 'rr_target': 1.0}),
        ("P4i: noSAL", {**p4_base, 'use_stop_after_loss': False}),
        ("P4j: OR", {**p4_base, 'use_opening_range_filter': True}),
        ("P4k: cd15", {**p4_base, 'cooldown_minutes': 15}),
        ("P4l: age15/noSAL", {**p4_base, 'max_fvg_age_m15': 15, 'use_stop_after_loss': False}),
        ("P4m: age15/trail25", {**p4_base, 'max_fvg_age_m15': 15, 'trail_offset_pct': 25}),
        ("P4n: 9:50/age15", {**p4_base, 'entry_start_time': '09:50', 'max_fvg_age_m15': 15}),
    ]

    print(f"Phase 4 fine-tuning: {len(configs)} variants...\n")
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
