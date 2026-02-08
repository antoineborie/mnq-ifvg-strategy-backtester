import itertools
import time
import pandas as pd
import numpy as np
from ifvg_strategy import IFVGStrategy


PARAM_GRID = {
    'min_fvg_size': [2.0, 3.0, 4.0, 6.0, 8.0],
    'max_fvg_age_m15': [6, 10, 14, 20],
    'rr_target': [2.0, 2.5, 3.0, 4.0],
    'max_trades_per_day': [1, 2, 3],
    'retracement_pct': [30, 50, 70],
    'min_risk_pts': [3.0, 5.0],
    'max_risk_pts': [30.0, 50.0],
    'be_trigger_rr': [1.0, 1.5, 2.0],
    'cooldown_minutes': [5, 10],
}

FIXED_PARAMS = {
    'killzone_start': '09:30',
    'killzone_end': '11:00',
    'use_be': True,
    'contract_value': 2.0,
    'target_mode': 'fixed_rr',
    'structure_lookback_days': 20,
}

QUICK_PARAM_GRID = {
    'min_fvg_size': [3.0, 4.0, 6.0],
    'max_fvg_age_m15': [8, 12, 18],
    'rr_target': [2.0, 3.0, 4.0],
    'max_trades_per_day': [1, 2],
    'retracement_pct': [40, 50, 60],
    'min_risk_pts': [3.0, 5.0],
    'max_risk_pts': [30.0, 50.0],
    'be_trigger_rr': [1.0, 1.5],
    'cooldown_minutes': [5, 10],
}


def _precompute_days(df_utc):
    if df_utc.index.tz is None:
        df = df_utc.tz_localize('UTC').tz_convert('America/New_York')
    else:
        df = df_utc.tz_convert('America/New_York')

    df = df.copy()
    df['_date'] = df.index.date
    grouped = {d: g for d, g in df.groupby('_date')}
    trading_days = sorted(grouped.keys())

    daily_ohlc = df.groupby('_date').agg(
        day_high=('high', 'max'),
        day_low=('low', 'min'),
    )

    precomputed = []

    for i, day in enumerate(trading_days):
        day_data = grouped[day]
        if len(day_data) < 30:
            continue

        killzone_m1 = day_data.between_time('09:30', '11:00')
        if len(killzone_m1) < 5:
            continue

        start_idx = max(0, i - 20)
        history_days = trading_days[start_idx:i]
        history_ohlc = daily_ohlc.loc[daily_ohlc.index.isin(history_days)] if history_days else pd.DataFrame()

        structure = _compute_structure_fast(history_ohlc, day_data)

        h1_bias = _determine_h1_bias(day_data, killzone_m1)
        if h1_bias is None:
            continue

        pre_kz = day_data[day_data.index < killzone_m1.index[0]]
        kz_and_before = pd.concat([pre_kz, killzone_m1]).sort_index()

        m15_data = kz_and_before.resample('15min').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna()

        if len(m15_data) < 4:
            continue

        m15_highs = m15_data['high'].values
        m15_lows = m15_data['low'].values
        m15_closes = m15_data['close'].values
        m15_opens = m15_data['open'].values
        m15_times = m15_data.index

        kz_highs = killzone_m1['high'].values
        kz_lows = killzone_m1['low'].values
        kz_closes = killzone_m1['close'].values
        kz_times = killzone_m1.index

        remaining_data = day_data[day_data.index > killzone_m1.index[0]]
        rem_highs = remaining_data['high'].values
        rem_lows = remaining_data['low'].values
        rem_closes = remaining_data['close'].values
        rem_times = remaining_data.index

        precomputed.append({
            'day': day,
            'h1_bias': h1_bias,
            'structure': structure,
            'm15_highs': m15_highs,
            'm15_lows': m15_lows,
            'm15_closes': m15_closes,
            'm15_opens': m15_opens,
            'm15_times': m15_times,
            'kz_highs': kz_highs,
            'kz_lows': kz_lows,
            'kz_closes': kz_closes,
            'kz_times': kz_times,
            'rem_highs': rem_highs,
            'rem_lows': rem_lows,
            'rem_closes': rem_closes,
            'rem_times': rem_times,
            'day_data': day_data,
        })

    return precomputed


def _compute_structure_fast(history_ohlc, day_data):
    levels = {
        'daily_highs': [], 'daily_lows': [],
        'weekly_highs': [], 'weekly_lows': [],
        'swing_highs': [], 'swing_lows': [],
    }
    if history_ohlc.empty:
        return levels

    highs = history_ohlc['day_high'].values
    lows = history_ohlc['day_low'].values
    levels['daily_highs'] = [float(h) for h in highs]
    levels['daily_lows'] = [float(l) for l in lows]

    n = len(highs)
    for ws in range(0, n, 5):
        we = min(ws + 5, n)
        levels['weekly_highs'].append(float(np.max(highs[ws:we])))
        levels['weekly_lows'].append(float(np.min(lows[ws:we])))

    for i in range(1, n - 1):
        if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
            levels['swing_highs'].append(float(highs[i]))
        if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
            levels['swing_lows'].append(float(lows[i]))

    current_price = float(day_data.iloc[-1]['close']) if not day_data.empty else 0
    for key in levels:
        levels[key] = sorted(
            [l for l in levels[key] if abs(l - current_price) < 500],
            key=lambda x: abs(x - current_price)
        )[:10]

    return levels


def _determine_h1_bias(day_data, killzone):
    if killzone.empty:
        return None
    kz_start = killzone.index[0]
    pre_kz = day_data[day_data.index < kz_start]
    if len(pre_kz) < 60:
        session_start = day_data.between_time('08:30', '09:30')
        if session_start.empty:
            return None
        pre_kz = session_start

    h1 = pre_kz.resample('1h').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
    }).dropna()

    if h1.empty:
        if not pre_kz.empty:
            o = pre_kz.iloc[0]['open']
            c = pre_kz.iloc[-1]['close']
            return 'BUY' if c > o else 'SELL'
        return None

    last_h1 = h1.iloc[-1]
    return 'BUY' if last_h1['close'] > last_h1['open'] else 'SELL'


def _run_fast_backtest(precomputed_days, config):
    trades = []
    min_fvg = config['min_fvg_size']
    max_age = config['max_fvg_age_m15']
    rr = config['rr_target']
    max_trades = config['max_trades_per_day']
    ret_pct = config['retracement_pct'] / 100.0
    min_risk = config['min_risk_pts']
    max_risk = config['max_risk_pts']
    use_be = config.get('use_be', True)
    be_trigger = config['be_trigger_rr']
    cooldown = config['cooldown_minutes']
    contract_val = config.get('contract_value', 2.0)
    target_mode = config.get('target_mode', 'fixed_rr')

    for pd_day in precomputed_days:
        h1_bias = pd_day['h1_bias']
        structure = pd_day['structure']
        m15_h = pd_day['m15_highs']
        m15_l = pd_day['m15_lows']
        m15_c = pd_day['m15_closes']
        m15_o = pd_day['m15_opens']
        m15_t = pd_day['m15_times']

        fvgs = []
        for i in range(2, len(m15_h)):
            gap_up = m15_l[i] - m15_h[i-2]
            if gap_up >= min_fvg and (m15_c[i-1] - m15_o[i-1]) > 0:
                fvgs.append(('bullish', float(m15_l[i]), float(m15_h[i-2]),
                             float((m15_l[i] + m15_h[i-2]) / 2), float(gap_up), m15_t[i], i))

            gap_down = m15_l[i-2] - m15_h[i]
            if gap_down >= min_fvg and (m15_c[i-1] - m15_o[i-1]) < 0:
                fvgs.append(('bearish', float(m15_l[i-2]), float(m15_h[i]),
                             float((m15_l[i-2] + m15_h[i]) / 2), float(gap_down), m15_t[i], i))

        ifvgs = []
        for ftype, top, bottom, midpoint, size, ftime, fidx in fvgs:
            later_mask = m15_t > ftime
            later_idx = np.where(later_mask)[0]
            if len(later_idx) == 0:
                continue
            later_idx = later_idx[:max_age]

            if ftype == 'bullish' and h1_bias == 'SELL':
                for j in later_idx:
                    if m15_c[j] < bottom:
                        ifvgs.append(('SELL', top, bottom, midpoint, size, m15_t[j],
                                      top, midpoint, fidx))
                        break
            elif ftype == 'bearish' and h1_bias == 'BUY':
                for j in later_idx:
                    if m15_c[j] > top:
                        ifvgs.append(('BUY', top, bottom, midpoint, size, m15_t[j],
                                      midpoint, bottom, fidx))
                        break

        kz_h = pd_day['kz_highs']
        kz_l = pd_day['kz_lows']
        kz_c = pd_day['kz_closes']
        kz_t = pd_day['kz_times']

        trades_today = 0
        last_trade_time = None
        used_ids = set()

        for i in range(len(kz_h)):
            if trades_today >= max_trades:
                break

            ct = kz_t[i]
            if last_trade_time is not None:
                if (ct - last_trade_time).total_seconds() / 60 < cooldown:
                    continue

            for ifvg_idx, ifvg in enumerate(ifvgs):
                direction, ftop, fbottom, fmid, fsize, inv_time, ztop, zbot, fid = ifvg
                if fid in used_ids:
                    continue
                if trades_today >= max_trades:
                    break
                if direction != h1_bias:
                    continue
                if ct <= inv_time:
                    continue

                zone_range = ztop - zbot
                if zone_range <= 0:
                    continue

                if direction == 'SELL':
                    entry_bot = ztop - (zone_range * ret_pct)
                    if kz_h[i] >= entry_bot and kz_c[i] < ztop:
                        entry_price = kz_c[i]
                        sl_price = ztop + 2.0
                        risk = sl_price - entry_price
                        if risk <= 0 or risk > max_risk or risk < min_risk:
                            continue
                        tp_price = entry_price - (risk * rr)
                    else:
                        continue
                else:
                    adj_top = zbot + (zone_range * ret_pct)
                    if kz_l[i] <= adj_top and kz_c[i] > zbot:
                        entry_price = kz_c[i]
                        sl_price = zbot - 2.0
                        risk = entry_price - sl_price
                        if risk <= 0 or risk > max_risk or risk < min_risk:
                            continue
                        tp_price = entry_price + (risk * rr)
                    else:
                        continue

                rem_h = pd_day['rem_highs']
                rem_l = pd_day['rem_lows']
                rem_c = pd_day['rem_closes']
                rem_t = pd_day['rem_times']

                start_j = 0
                for sj in range(len(rem_t)):
                    if rem_t[sj] > ct:
                        start_j = sj
                        break

                result_label, exit_price, exit_time, pnl_pts = _sim_trade_fast(
                    rem_h, rem_l, rem_c, rem_t, start_j,
                    entry_price, tp_price, sl_price, direction, risk,
                    use_be, be_trigger
                )

                trades.append({
                    'entry_time': ct,
                    'exit_time': exit_time,
                    'direction': direction,
                    'entry': round(entry_price, 2),
                    'sl': round(sl_price, 2),
                    'tp': round(tp_price, 2),
                    'exit_price': round(exit_price, 2),
                    'risk_pts': round(risk, 2),
                    'pnl_pts': round(pnl_pts, 2),
                    'pnl_dollars': round(pnl_pts * contract_val, 2),
                    'result': result_label,
                    'rr_achieved': round(pnl_pts / risk, 2) if risk > 0 else 0,
                    'fvg_size': round(fsize, 2),
                    'h1_bias': direction,
                    'target_mode': target_mode,
                })
                trades_today += 1
                last_trade_time = ct
                used_ids.add(fid)
                break

    return trades


def _sim_trade_fast(highs, lows, closes, times, start, entry, tp, sl, direction, risk, use_be, be_trigger):
    current_sl = sl
    be_activated = False
    be_level = entry + (risk * be_trigger) if direction == 'BUY' else entry - (risk * be_trigger)

    for i in range(start, len(highs)):
        h, l, c = highs[i], lows[i], closes[i]

        if direction == 'BUY':
            if l <= current_sl:
                pnl = current_sl - entry
                label = 'BE' if be_activated and abs(pnl) < 2 else ('WIN' if pnl > 0 else 'LOSS')
                return label, current_sl, times[i], pnl
            if h >= tp:
                return 'WIN', tp, times[i], tp - entry
            if use_be and not be_activated and h >= be_level:
                current_sl = entry + 1.0
                be_activated = True
        else:
            if h >= current_sl:
                pnl = entry - current_sl
                label = 'BE' if be_activated and abs(pnl) < 2 else ('WIN' if pnl > 0 else 'LOSS')
                return label, current_sl, times[i], pnl
            if l <= tp:
                return 'WIN', tp, times[i], entry - tp
            if use_be and not be_activated and l <= be_level:
                current_sl = entry - 1.0
                be_activated = True

    if len(closes) > 0:
        last_c = closes[-1]
        pnl = (last_c - entry) if direction == 'BUY' else (entry - last_c)
        return 'EOD', last_c, times[-1], pnl
    return 'EOD', entry, times[start] if start < len(times) else None, 0


def _build_metrics(trades_list, contract_val=2.0):
    if not trades_list:
        return {
            'total_trades': 0, 'wins': 0, 'losses': 0, 'breakevens': 0,
            'win_rate': 0, 'total_pnl_pts': 0, 'total_pnl_dollars': 0,
            'profit_factor': 0, 'max_drawdown_pts': 0, 'avg_rr_on_wins': 0,
            'trades_per_month': 0, 'avg_daily_pnl': 0,
            'max_consecutive_losses': 0, 'winning_days': 0,
            'total_trading_days': 0, 'calmar_ratio': 0,
        }

    pnls = np.array([t['pnl_pts'] for t in trades_list])
    results = [t['result'] for t in trades_list]
    total = len(pnls)
    wins = sum(1 for r in results if r == 'WIN')
    losses = sum(1 for r in results if r == 'LOSS')
    bes = sum(1 for r in results if r == 'BE')
    win_rate = (wins / total * 100) if total > 0 else 0
    total_pnl = float(pnls.sum())

    gross_profit = float(pnls[pnls > 0].sum())
    gross_loss = float(abs(pnls[pnls < 0].sum()))
    pf = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')
    if pf == float('inf'):
        pf = 99.0

    cum = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    max_dd = float(dd.min())

    win_rrs = [t['rr_achieved'] for t in trades_list if t['result'] == 'WIN']
    avg_rr = np.mean(win_rrs) if win_rrs else 0

    dates = set()
    daily_pnl = {}
    for t in trades_list:
        d = t['entry_time'].date() if hasattr(t['entry_time'], 'date') else t['entry_time']
        dates.add(d)
        daily_pnl[d] = daily_pnl.get(d, 0) + t['pnl_pts']

    total_days = len(dates)
    winning_days = sum(1 for v in daily_pnl.values() if v > 0)
    avg_daily = np.mean(list(daily_pnl.values())) if daily_pnl else 0
    trades_per_month = total / max(total_days / 21, 1)

    max_cons_losses = 0
    current_streak = 0
    for p in pnls:
        if p < 0:
            current_streak += 1
            max_cons_losses = max(max_cons_losses, current_streak)
        else:
            current_streak = 0

    calmar = total_pnl / abs(max_dd) if max_dd != 0 else 0

    return {
        'total_trades': total,
        'wins': wins,
        'losses': losses,
        'breakevens': bes,
        'win_rate': round(win_rate, 1),
        'total_pnl_pts': round(total_pnl, 2),
        'total_pnl_dollars': round(total_pnl * contract_val, 2),
        'profit_factor': round(pf, 2),
        'max_drawdown_pts': round(max_dd, 2),
        'avg_rr_on_wins': round(avg_rr, 2),
        'trades_per_month': round(trades_per_month, 1),
        'avg_daily_pnl': round(avg_daily, 2),
        'max_consecutive_losses': max_cons_losses,
        'winning_days': winning_days,
        'total_trading_days': total_days,
        'calmar_ratio': round(calmar, 2),
    }


def _compute_score(metrics):
    pnl = metrics['total_pnl_pts']
    pf = metrics['profit_factor']
    dd = abs(metrics['max_drawdown_pts']) if metrics['max_drawdown_pts'] != 0 else 1
    trades = metrics['total_trades']
    wr = metrics['win_rate']

    if trades < 20:
        return -9999

    score = 0
    score += pnl * 0.3
    score += min(pf, 5) * 50
    calmar = pnl / dd if dd > 0 else 0
    score += calmar * 30
    score += wr * 1.5
    if trades >= 50:
        score += 20
    if metrics['max_consecutive_losses'] <= 5:
        score += 15

    return round(score, 2)


def run_optimization(df, param_grid=None, fixed_params=None, top_n=20,
                     progress_callback=None, max_combos=None):
    if param_grid is None:
        param_grid = PARAM_GRID
    if fixed_params is None:
        fixed_params = FIXED_PARAMS

    if progress_callback:
        progress_callback(0, 1, 0, 0, "Pre-processing data...")

    precomputed = _precompute_days(df)

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    all_combos = list(itertools.product(*values))

    if max_combos and len(all_combos) > max_combos:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(all_combos), size=max_combos, replace=False)
        all_combos = [all_combos[i] for i in sorted(indices)]

    total = len(all_combos)
    results = []
    t0 = time.time()

    for idx, combo in enumerate(all_combos):
        config = dict(zip(keys, combo))
        config.update(fixed_params)

        trades_list = _run_fast_backtest(precomputed, config)
        m = _build_metrics(trades_list, config.get('contract_value', 2.0))

        row = {**config}
        row.update(m)
        row['score'] = _compute_score(m)
        results.append(row)

        if progress_callback and (idx % 50 == 0 or idx == total - 1):
            elapsed = time.time() - t0
            eta = (elapsed / (idx + 1)) * (total - idx - 1) if idx > 0 else 0
            progress_callback(idx + 1, total, elapsed, eta)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('score', ascending=False).reset_index(drop=True)

    elapsed_total = time.time() - t0

    return {
        'results': results_df,
        'total_combos': total,
        'elapsed': round(elapsed_total, 1),
        'top_n': results_df.head(top_n),
    }


def get_param_grid_size(param_grid=None):
    if param_grid is None:
        param_grid = PARAM_GRID
    total = 1
    for v in param_grid.values():
        total *= len(v)
    return total
