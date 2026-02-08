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
          f"TPW={tpw:.1f} B60={below_60}/{total_q} CONS={consistency:.0f}% B50={below_50} MLS={max_ls}")

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
        'tpw': tpw, 'below_50': below_50, 'max_ls': max_ls,
    }

base = {
    'min_fvg_size': 3.0, 'max_fvg_age_m15': 15, 'rr_target': 1.2,
    'max_risk_pts': 25.0, 'min_risk_pts': 5.0, 'max_trades_per_day': 1,
    'killzone_start': '09:30', 'killzone_end': '11:00',
    'use_be': True, 'be_trigger_rr': 0.5, 'cooldown_minutes': 10,
    'contract_value': 2.0, 'target_mode': 'fixed_rr', 'retracement_pct': 60,
    'structure_lookback_days': 20, 'use_displacement_filter': True,
    'min_displacement_body_pct': 55, 'min_displacement_size': 3.5,
    'use_m1_confirmation': True, 'use_trailing_stop': True,
    'trail_trigger_rr': 0.5, 'trail_offset_pct': 30,
    'entry_start_time': '10:05',
    'use_stop_after_loss': True,
    'use_opening_range_filter': True,
}

print("=" * 80)
print("ITERATION 5: REFINING OR+1TPD (B60=4, CONS=87%)")
print("=" * 80)

print("\n--- Baseline OR+1TPD ---")
test_config("OR+1TPD", base, verbose=True)

print("\n--- Vary EST ---")
for est in ['10:00', '10:05', '10:10', '10:15']:
    test_config(f"EST={est}", {**base, 'entry_start_time': est})

print("\n--- Vary FVG age ---")
for age in [8, 10, 12, 15]:
    test_config(f"AGE={age}", {**base, 'max_fvg_age_m15': age})

print("\n--- Vary disp body ---")
for body in [50, 55, 60, 65, 70]:
    test_config(f"BODY={body}", {**base, 'min_displacement_body_pct': body})

print("\n--- Vary disp size ---")
for ds in [3.0, 3.5, 4.0, 4.5]:
    test_config(f"DSIZE={ds}", {**base, 'min_displacement_size': ds})

print("\n--- Vary FVG size ---")
for fvg in [3.0, 3.5, 4.0, 4.5]:
    test_config(f"FVG={fvg}", {**base, 'min_fvg_size': fvg})

print("\n--- Vary retracement ---")
for ret in [40, 50, 55, 60, 70]:
    test_config(f"RET={ret}", {**base, 'retracement_pct': ret})

print("\n--- Vary max risk ---")
for mr in [15, 18, 20, 25]:
    test_config(f"RISK={mr}", {**base, 'max_risk_pts': mr})

print("\n--- Extended killzone ---")
for ke in ['11:00', '11:30']:
    test_config(f"KZ_END={ke}", {**base, 'killzone_end': ke})

print("\n--- Enhanced bias + OR + 1TPD ---")
for conf in [2, 3]:
    test_config(f"BIAS={conf}", {**base, 'use_enhanced_bias': True, 'bias_min_confidence': conf})

print("\n--- Best combinations ---")
combos = []
for age in [8, 10, 12]:
    for body in [55, 60, 65]:
        for ret in [50, 60]:
            for fvg in [3.0, 4.0]:
                cfg = {**base, 'max_fvg_age_m15': age,
                       'min_displacement_body_pct': body,
                       'retracement_pct': ret,
                       'min_fvg_size': fvg}
                r = test_config(f"A{age}/B{body}/R{ret}/F{fvg}", cfg)
                if r:
                    combos.append(r)

combos.sort(key=lambda x: (-x['consistency'], x['below_60'], -x['wr']))
print(f"\n--- TOP 10 COMBOS ---")
for i, r in enumerate(combos[:10]):
    print(f"  {i+1}. {r['name']}: {r['trades']}t WR={r['wr']:.1f}% P&L={r['pnl']:+.0f} "
          f"TPW={r['tpw']:.1f} B60={r['below_60']}/{r['total_q']} CONS={r['consistency']:.0f}%")

if combos and combos[0]['below_60'] <= 3:
    print(f"\n--- BEST COMBO DETAIL ---")
    c = combos[0]
    parts = c['name'].split('/')
    age = int(parts[0][1:])
    body = int(parts[1][1:])
    ret = int(parts[2][1:])
    fvg = float(parts[3][1:])
    cfg = {**base, 'max_fvg_age_m15': age, 'min_displacement_body_pct': body,
           'retracement_pct': ret, 'min_fvg_size': fvg}
    test_config(f"BEST COMBO: {c['name']}", cfg, verbose=True)
