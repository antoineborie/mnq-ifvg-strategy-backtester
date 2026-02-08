import sys
sys.path.insert(0, '.')
from data_loader import load_data, get_active_contract, list_data_files
from ifvg_strategy import IFVGStrategy
import pandas as pd
import numpy as np

files = list_data_files()
raw_df = load_data(files)
df = get_active_contract(raw_df)

def test_config(name, config):
    strategy = IFVGStrategy(config)
    results = strategy.run_backtest(df)
    tdf = results['trades']
    if isinstance(tdf, list):
        tdf = pd.DataFrame(tdf)
    if tdf is None or len(tdf) == 0:
        print(f"\n{name}: NO TRADES")
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
    avg_win = tdf[tdf['result'] == 'WIN']['pnl_pts'].mean() if (tdf['result'] == 'WIN').any() else 0
    avg_loss = tdf[tdf['result'] == 'LOSS']['pnl_pts'].mean() if (tdf['result'] == 'LOSS').any() else 0

    first_date = tdf['et_naive'].min()
    last_date = tdf['et_naive'].max()
    weeks = max(1, (last_date - first_date).days / 7)
    tpw = total_trades / weeks

    print(f"\n=== {name} ===")
    print(f"Trades: {total_trades} | WR: {overall_wr:.1f}% | P&L: {total_pnl:+.1f} pts | Trades/wk: {tpw:.1f}")
    print(f"Avg WIN: {avg_win:+.1f} | Avg LOSS: {avg_loss:+.1f}")
    print(f"Qualified months: {total_q} | Below 60%: {below_60} | Below 50%: {below_50}")
    print(f"CONSISTENCY: {consistency:.1f}% of months at 60%+")

    weak = qualified[qualified['wr'] < 60]
    if len(weak) > 0:
        print(f"Weak months: ", end="")
        for _, row in weak.iterrows():
            print(f"{row['year_month']}({row['wr']:.0f}%/{row['trades']}t) ", end="")
        print()

    return {
        'trades': total_trades, 'wr': overall_wr, 'pnl': total_pnl,
        'below_60': below_60, 'total_q': total_q, 'consistency': consistency,
        'tpw': tpw, 'below_50': below_50,
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
    'trail_trigger_rr': 0.5, 'trail_offset_pct': 30, 'entry_start_time': '09:45',
}

print("=" * 70)
print("TESTING STRATEGY IMPROVEMENTS FOR MONTHLY CONSISTENCY")
print("=" * 70)

test_config("BASELINE", base)

cfg1 = {**base, 'entry_start_time': '10:00'}
test_config("DELAYED ENTRY (10:00)", cfg1)

cfg2 = {**base, 'use_enhanced_bias': True, 'bias_min_confidence': 2}
test_config("ENHANCED BIAS (conf>=2)", cfg2)

cfg3 = {**base, 'use_enhanced_bias': True, 'bias_min_confidence': 3}
test_config("ENHANCED BIAS (conf>=3)", cfg3)

cfg4 = {**base, 'use_stop_after_loss': True}
test_config("STOP AFTER LOSS", cfg4)

cfg5 = {**base, 'entry_start_time': '10:00', 'use_stop_after_loss': True}
test_config("DELAYED + STOP AFTER LOSS", cfg5)

cfg6 = {**base, 'entry_start_time': '10:00', 'use_enhanced_bias': True,
         'bias_min_confidence': 2, 'use_stop_after_loss': True}
test_config("DELAYED + BIAS + STOP", cfg6)

cfg7 = {**base, 'entry_start_time': '10:00', 'use_enhanced_bias': True,
         'bias_min_confidence': 2, 'use_stop_after_loss': True,
         'rr_target': 1.0}
test_config("DELAYED+BIAS+STOP+RR1.0", cfg7)

cfg8 = {**base, 'entry_start_time': '10:00', 'use_enhanced_bias': True,
         'bias_min_confidence': 2, 'use_stop_after_loss': True,
         'rr_target': 1.0, 'trail_trigger_rr': 0.7, 'trail_offset_pct': 40}
test_config("ALL+SOFTER TRAIL", cfg8)

cfg9 = {**base, 'entry_start_time': '10:00', 'use_enhanced_bias': True,
         'bias_min_confidence': 2, 'use_stop_after_loss': True,
         'rr_target': 1.0, 'use_trailing_stop': False}
test_config("ALL+NO TRAIL", cfg9)

cfg10 = {**base, 'entry_start_time': '10:00', 'use_enhanced_bias': True,
          'bias_min_confidence': 2, 'use_stop_after_loss': True,
          'rr_target': 1.0, 'use_trailing_stop': False,
          'use_opening_range_filter': True}
test_config("ALL+NO TRAIL+OR FILTER", cfg10)

cfg11 = {**base, 'entry_start_time': '10:00', 'use_enhanced_bias': True,
          'bias_min_confidence': 2, 'use_stop_after_loss': True,
          'rr_target': 1.0, 'use_trailing_stop': False,
          'max_trades_per_day': 1}
test_config("ALL+NO TRAIL+1 TRADE/DAY", cfg11)

cfg12 = {**base, 'entry_start_time': '10:00', 'use_enhanced_bias': True,
          'bias_min_confidence': 3, 'use_stop_after_loss': True,
          'rr_target': 1.0, 'use_trailing_stop': False}
test_config("STRICT BIAS(3)+ALL", cfg12)

cfg13 = {**base, 'entry_start_time': '10:00', 'use_enhanced_bias': True,
          'bias_min_confidence': 2, 'use_stop_after_loss': True,
          'rr_target': 1.0, 'use_trailing_stop': False,
          'use_multi_confirm': True, 'use_m1_confirmation': False,
          'confirm_require_body_ratio': 40}
test_config("ENHANCED CONFIRM+ALL", cfg13)
