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
        print(f"  {name}: NO TRADES")
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
          f"B50={below_50} MXLS={max_loss_streak}")

    if verbose:
        for _, row in monthly.iterrows():
            flag = ""
            if row['trades'] >= 3:
                if row['wr'] < 50: flag = " <<BAD"
                elif row['wr'] < 60: flag = " <WEAK"
            print(f"    {row['year_month']}: {row['trades']}t WR={row['wr']:.0f}% P&L={row['pnl']:+.1f}{flag}")

    return {
        'trades': total_trades, 'wr': overall_wr, 'pnl': total_pnl,
        'below_60': below_60, 'total_q': total_q, 'consistency': consistency,
        'tpw': tpw, 'below_50': below_50, 'max_loss_streak': max_loss_streak,
    }

best_base = {
    'min_fvg_size': 3.0, 'max_fvg_age_m15': 15, 'rr_target': 1.2,
    'max_risk_pts': 25.0, 'min_risk_pts': 5.0, 'max_trades_per_day': 2,
    'killzone_start': '09:30', 'killzone_end': '11:00',
    'use_be': True, 'be_trigger_rr': 0.5, 'cooldown_minutes': 10,
    'contract_value': 2.0, 'target_mode': 'fixed_rr', 'retracement_pct': 60,
    'structure_lookback_days': 20, 'use_displacement_filter': True,
    'min_displacement_body_pct': 55, 'min_displacement_size': 3.5,
    'use_m1_confirmation': True, 'use_trailing_stop': True,
    'trail_trigger_rr': 0.5, 'trail_offset_pct': 30,
    'entry_start_time': '10:00',
    'use_stop_after_loss': True,
}

print("=" * 80)
print("ITERATION 2: REFINING BEST CONFIG (DELAYED+STOP AFTER LOSS)")
print("=" * 80)

print("\n--- Current best baseline ---")
test_config("BEST_BASE", best_base, verbose=True)

print("\n--- Vary entry start time ---")
for est in ['09:50', '09:55', '10:00', '10:05', '10:10']:
    test_config(f"EST={est}", {**best_base, 'entry_start_time': est})

print("\n--- Vary RR target ---")
for rr in [0.8, 0.9, 1.0, 1.1, 1.2, 1.5]:
    test_config(f"RR={rr}", {**best_base, 'rr_target': rr})

print("\n--- Vary min displacement ---")
for disp in [3.0, 3.5, 4.0, 4.5, 5.0]:
    test_config(f"DISP_SIZE={disp}", {**best_base, 'min_displacement_size': disp})

print("\n--- Vary displacement body % ---")
for body in [45, 50, 55, 60, 65]:
    test_config(f"DISP_BODY={body}%", {**best_base, 'min_displacement_body_pct': body})

print("\n--- Vary min FVG size ---")
for fvg in [3.0, 3.5, 4.0, 5.0]:
    test_config(f"FVG_SIZE={fvg}", {**best_base, 'min_fvg_size': fvg})

print("\n--- Vary trail params ---")
for tt, to in [(0.5, 30), (0.6, 30), (0.7, 30), (0.5, 40), (0.6, 40), (0.7, 40), (0.8, 50)]:
    test_config(f"TRAIL={tt}/{to}%", {**best_base, 'trail_trigger_rr': tt, 'trail_offset_pct': to})

print("\n--- Vary BE trigger ---")
for be in [0.4, 0.5, 0.6, 0.7, 0.8]:
    test_config(f"BE={be}", {**best_base, 'be_trigger_rr': be})

print("\n--- Vary max risk ---")
for mr in [15, 18, 20, 25, 30]:
    test_config(f"MAXRISK={mr}", {**best_base, 'max_risk_pts': mr})

print("\n--- Vary max FVG age ---")
for age in [8, 10, 12, 15, 20]:
    test_config(f"AGE={age}", {**best_base, 'max_fvg_age_m15': age})

print("\n--- Vary retracement ---")
for ret in [40, 50, 55, 60, 70]:
    test_config(f"RET={ret}%", {**best_base, 'retracement_pct': ret})

print("\n--- Vary cooldown ---")
for cd in [5, 10, 15, 20, 30]:
    test_config(f"CD={cd}min", {**best_base, 'cooldown_minutes': cd})

print("\n--- Vary max trades/day ---")
for mt in [1, 2, 3]:
    test_config(f"MT={mt}", {**best_base, 'max_trades_per_day': mt})

print("\n--- Combined optimizations ---")
for rr in [1.0, 1.1, 1.2]:
    for disp in [3.5, 4.0]:
        for body in [50, 55]:
            for fvg in [3.0, 4.0]:
                for be in [0.5, 0.6]:
                    for age in [10, 12, 15]:
                        cfg = {**best_base, 'rr_target': rr, 'min_displacement_size': disp,
                               'min_displacement_body_pct': body, 'min_fvg_size': fvg,
                               'be_trigger_rr': be, 'max_fvg_age_m15': age}
                        r = test_config(
                            f"RR={rr}/D={disp}/B={body}/F={fvg}/BE={be}/A={age}", cfg)
                        if r and r['below_60'] <= 5:
                            print(f"    *** PROMISING: {r['below_60']} months below 60%! ***")
                            test_config(f"  DETAIL", cfg, verbose=True)
