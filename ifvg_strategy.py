import pandas as pd
import numpy as np


class IFVGStrategy:
    def __init__(self, config=None):
        defaults = {
            'min_fvg_size': 4.0,
            'max_fvg_age_m15': 12,
            'rr_target': 3.0,
            'max_risk_pts': 50.0,
            'min_risk_pts': 5.0,
            'max_trades_per_day': 2,
            'killzone_start': '09:30',
            'killzone_end': '11:00',
            'use_be': True,
            'be_trigger_rr': 1.5,
            'cooldown_minutes': 10,
            'contract_value': 2.0,
            'target_mode': 'fixed_rr',
            'retracement_pct': 50,
            'structure_lookback_days': 20,
        }
        self.config = {**defaults, **(config or {})}
        self.trades = []

    def run_backtest(self, df_utc):
        self.trades = []

        if df_utc.index.tz is None:
            df = df_utc.tz_localize('UTC').tz_convert('America/New_York')
        else:
            df = df_utc.tz_convert('America/New_York')

        trading_days = sorted(set(df.index.date))

        for day in trading_days:
            day_data = df[df.index.date == day]
            if len(day_data) < 30:
                continue

            lookback_start = day - pd.Timedelta(days=self.config['structure_lookback_days'])
            history = df[(df.index.date >= lookback_start) & (df.index.date < day)]

            self._process_day(day_data, history, day, df)

        return self._build_results()

    def _process_day(self, day_data, history, day, full_df):
        ks = self.config['killzone_start']
        ke = self.config['killzone_end']

        killzone_m1 = day_data.between_time(ks, ke)
        if len(killzone_m1) < 5:
            return

        structure = self._compute_structure_levels(history, day_data)

        h1_bias = self._determine_h1_bias(day_data, killzone_m1)
        if h1_bias is None:
            return

        pre_kz = day_data[day_data.index < killzone_m1.index[0]]
        kz_and_before = pd.concat([pre_kz, killzone_m1]).sort_index()

        m15_data = kz_and_before.resample('15min').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna()

        if len(m15_data) < 4:
            return

        m15_fvgs = self._detect_fvgs_m15(m15_data)

        m15_ifvgs = self._detect_ifvgs_m15(m15_fvgs, m15_data, h1_bias, structure)

        trades_today = 0
        max_trades = self.config['max_trades_per_day']
        last_trade_time = None
        cooldown = self.config['cooldown_minutes']
        used_ifvgs = set()

        for i in range(len(killzone_m1)):
            if trades_today >= max_trades:
                break

            current_time = killzone_m1.index[i]
            current_bar = killzone_m1.iloc[i]

            if last_trade_time is not None:
                minutes_since = (current_time - last_trade_time).total_seconds() / 60
                if minutes_since < cooldown:
                    continue

            for ifvg in m15_ifvgs:
                if ifvg['id'] in used_ifvgs:
                    continue
                if trades_today >= max_trades:
                    break

                if ifvg['direction'] != h1_bias:
                    continue

                entry_signal = self._check_m1_retracement(
                    ifvg, current_bar, current_time
                )

                if entry_signal:
                    remaining = day_data[day_data.index > current_time]
                    trade = self._execute_trade(
                        entry_signal, current_bar, current_time, remaining, structure
                    )
                    if trade:
                        self.trades.append(trade)
                        trades_today += 1
                        last_trade_time = current_time
                        used_ifvgs.add(ifvg['id'])
                        break

    def _compute_structure_levels(self, history, day_data):
        levels = {
            'daily_highs': [],
            'daily_lows': [],
            'weekly_highs': [],
            'weekly_lows': [],
            'swing_highs': [],
            'swing_lows': [],
        }

        if history.empty:
            return levels

        daily = history.resample('D').agg({'high': 'max', 'low': 'min'}).dropna()
        if not daily.empty:
            for _, row in daily.iterrows():
                levels['daily_highs'].append(float(row['high']))
                levels['daily_lows'].append(float(row['low']))

        weekly = history.resample('W').agg({'high': 'max', 'low': 'min'}).dropna()
        if not weekly.empty:
            for _, row in weekly.iterrows():
                levels['weekly_highs'].append(float(row['high']))
                levels['weekly_lows'].append(float(row['low']))

        if len(daily) >= 3:
            highs = daily['high'].values
            lows = daily['low'].values
            for i in range(1, len(daily) - 1):
                if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                    levels['swing_highs'].append(float(highs[i]))
                if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                    levels['swing_lows'].append(float(lows[i]))

        if day_data is not None and not day_data.empty:
            current_price = float(day_data.iloc[-1]['close'])
            for key in ['daily_highs', 'daily_lows', 'weekly_highs', 'weekly_lows',
                         'swing_highs', 'swing_lows']:
                levels[key] = sorted(
                    [l for l in levels[key] if abs(l - current_price) < 500],
                    key=lambda x: abs(x - current_price)
                )[:10]

        if levels['daily_highs']:
            levels['pdh'] = levels['daily_highs'][-1] if levels['daily_highs'] else None
        if levels['daily_lows']:
            levels['pdl'] = levels['daily_lows'][-1] if levels['daily_lows'] else None

        return levels

    def _determine_h1_bias(self, day_data, killzone):
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
        if last_h1['close'] > last_h1['open']:
            return 'BUY'
        else:
            return 'SELL'

    def _detect_fvgs_m15(self, m15_data):
        fvgs = []
        if len(m15_data) < 3:
            return fvgs

        highs = m15_data['high'].values
        lows = m15_data['low'].values
        closes = m15_data['close'].values
        opens = m15_data['open'].values
        times = m15_data.index
        min_size = self.config['min_fvg_size']

        for i in range(2, len(m15_data)):
            gap_up = lows[i] - highs[i - 2]
            if gap_up >= min_size:
                mid_body = closes[i - 1] - opens[i - 1]
                if mid_body > 0:
                    fvgs.append({
                        'id': f"m15_bull_{times[i]}",
                        'type': 'bullish',
                        'top': float(lows[i]),
                        'bottom': float(highs[i - 2]),
                        'midpoint': float((lows[i] + highs[i - 2]) / 2),
                        'size': float(gap_up),
                        'time': times[i],
                        'filled': False,
                    })

            gap_down = lows[i - 2] - highs[i]
            if gap_down >= min_size:
                mid_body = closes[i - 1] - opens[i - 1]
                if mid_body < 0:
                    fvgs.append({
                        'id': f"m15_bear_{times[i]}",
                        'type': 'bearish',
                        'top': float(lows[i - 2]),
                        'bottom': float(highs[i]),
                        'midpoint': float((lows[i - 2] + highs[i]) / 2),
                        'size': float(gap_down),
                        'time': times[i],
                        'filled': False,
                    })

        return fvgs

    def _detect_ifvgs_m15(self, fvgs, m15_data, h1_bias, structure):
        ifvgs = []
        max_age = self.config['max_fvg_age_m15']

        if len(m15_data) < 2:
            return ifvgs

        for fvg in fvgs:
            if fvg['filled']:
                continue

            later_bars = m15_data[m15_data.index > fvg['time']]
            if later_bars.empty:
                continue

            age_bars = len(later_bars)
            if age_bars > max_age:
                continue

            for j in range(len(later_bars)):
                bar = later_bars.iloc[j]

                if fvg['type'] == 'bullish' and h1_bias == 'SELL':
                    if bar['close'] < fvg['bottom']:
                        ifvgs.append({
                            'id': fvg['id'],
                            'direction': 'SELL',
                            'fvg_top': fvg['top'],
                            'fvg_bottom': fvg['bottom'],
                            'fvg_midpoint': fvg['midpoint'],
                            'fvg_size': fvg['size'],
                            'inversion_time': later_bars.index[j],
                            'zone_top': fvg['top'],
                            'zone_bottom': fvg['midpoint'],
                        })
                        fvg['filled'] = True
                        break

                elif fvg['type'] == 'bearish' and h1_bias == 'BUY':
                    if bar['close'] > fvg['top']:
                        ifvgs.append({
                            'id': fvg['id'],
                            'direction': 'BUY',
                            'fvg_top': fvg['top'],
                            'fvg_bottom': fvg['bottom'],
                            'fvg_midpoint': fvg['midpoint'],
                            'fvg_size': fvg['size'],
                            'inversion_time': later_bars.index[j],
                            'zone_top': fvg['midpoint'],
                            'zone_bottom': fvg['bottom'],
                        })
                        fvg['filled'] = True
                        break

        return ifvgs

    def _check_m1_retracement(self, ifvg, bar, current_time):
        if current_time <= ifvg['inversion_time']:
            return None

        ret_pct = self.config['retracement_pct'] / 100.0
        zone_top = ifvg['zone_top']
        zone_bottom = ifvg['zone_bottom']
        zone_range = zone_top - zone_bottom

        if zone_range <= 0:
            return None

        entry_level_top = zone_top
        entry_level_bottom = zone_top - (zone_range * ret_pct)

        if ifvg['direction'] == 'SELL':
            if bar['high'] >= entry_level_bottom and bar['close'] < entry_level_top:
                return {
                    'direction': 'SELL',
                    'ifvg_id': ifvg['id'],
                    'fvg_top': ifvg['fvg_top'],
                    'fvg_bottom': ifvg['fvg_bottom'],
                    'fvg_midpoint': ifvg['fvg_midpoint'],
                    'fvg_size': ifvg['fvg_size'],
                    'zone_top': zone_top,
                    'zone_bottom': zone_bottom,
                    'inversion_time': ifvg['inversion_time'],
                }

        elif ifvg['direction'] == 'BUY':
            adjusted_bottom = zone_bottom
            adjusted_top = zone_bottom + (zone_range * ret_pct)

            if bar['low'] <= adjusted_top and bar['close'] > adjusted_bottom:
                return {
                    'direction': 'BUY',
                    'ifvg_id': ifvg['id'],
                    'fvg_top': ifvg['fvg_top'],
                    'fvg_bottom': ifvg['fvg_bottom'],
                    'fvg_midpoint': ifvg['fvg_midpoint'],
                    'fvg_size': ifvg['fvg_size'],
                    'zone_top': zone_top,
                    'zone_bottom': zone_bottom,
                    'inversion_time': ifvg['inversion_time'],
                }

        return None

    def _execute_trade(self, signal, entry_bar, entry_time, future_data, structure):
        if future_data.empty:
            return None

        direction = signal['direction']
        rr = self.config['rr_target']
        max_risk = self.config['max_risk_pts']
        min_risk = self.config['min_risk_pts']
        use_be = self.config['use_be']
        be_trigger = self.config['be_trigger_rr']
        contract_val = self.config['contract_value']
        target_mode = self.config['target_mode']

        entry_price = entry_bar['close']

        if direction == 'SELL':
            sl_price = signal['zone_top'] + 2.0
            risk = sl_price - entry_price
            if risk <= 0 or risk > max_risk or risk < min_risk:
                return None

            if target_mode == 'ssl':
                tp_price = self._find_ssl_target(entry_price, direction, structure, risk)
            else:
                tp_price = entry_price - (risk * rr)
        else:
            sl_price = signal['zone_bottom'] - 2.0
            risk = entry_price - sl_price
            if risk <= 0 or risk > max_risk or risk < min_risk:
                return None

            if target_mode == 'ssl':
                tp_price = self._find_ssl_target(entry_price, direction, structure, risk)
            else:
                tp_price = entry_price + (risk * rr)

        if tp_price is None:
            return None

        result, exit_price, exit_time, pnl_pts = self._simulate_trade(
            future_data, entry_price, tp_price, sl_price, direction, risk, use_be, be_trigger
        )

        pnl_dollars = pnl_pts * contract_val

        return {
            'entry_time': entry_time,
            'exit_time': exit_time,
            'direction': direction,
            'entry': round(entry_price, 2),
            'sl': round(sl_price, 2),
            'tp': round(tp_price, 2),
            'exit_price': round(exit_price, 2),
            'risk_pts': round(risk, 2),
            'pnl_pts': round(pnl_pts, 2),
            'pnl_dollars': round(pnl_dollars, 2),
            'result': result,
            'rr_achieved': round(pnl_pts / risk, 2) if risk > 0 else 0,
            'fvg_size': round(signal['fvg_size'], 2),
            'h1_bias': direction,
            'target_mode': target_mode,
        }

    def _find_ssl_target(self, entry_price, direction, structure, risk):
        min_target_distance = risk * 1.5

        if direction == 'SELL':
            candidates = []
            for key in ['swing_lows', 'daily_lows']:
                for level in structure.get(key, []):
                    dist = entry_price - level
                    if dist >= min_target_distance:
                        candidates.append(level)

            if candidates:
                return max(candidates)
            return entry_price - (risk * self.config['rr_target'])

        else:
            candidates = []
            for key in ['swing_highs', 'daily_highs']:
                for level in structure.get(key, []):
                    dist = level - entry_price
                    if dist >= min_target_distance:
                        candidates.append(level)

            if candidates:
                return min(candidates)
            return entry_price + (risk * self.config['rr_target'])

    def _simulate_trade(self, future_data, entry, tp, sl, direction, risk, use_be, be_trigger):
        current_sl = sl
        be_activated = False

        for ts, row in future_data.iterrows():
            h, l, c = row['high'], row['low'], row['close']

            if direction == 'BUY':
                if l <= current_sl:
                    pnl = current_sl - entry
                    label = 'BE' if be_activated and abs(pnl) < 2 else ('WIN' if pnl > 0 else 'LOSS')
                    return label, current_sl, ts, pnl
                if h >= tp:
                    return 'WIN', tp, ts, tp - entry

                if use_be and not be_activated:
                    if h >= entry + (risk * be_trigger):
                        current_sl = entry + 1.0
                        be_activated = True

            else:
                if h >= current_sl:
                    pnl = entry - current_sl
                    label = 'BE' if be_activated and abs(pnl) < 2 else ('WIN' if pnl > 0 else 'LOSS')
                    return label, current_sl, ts, pnl
                if l <= tp:
                    return 'WIN', tp, ts, entry - tp

                if use_be and not be_activated:
                    if l <= entry - (risk * be_trigger):
                        current_sl = entry - 1.0
                        be_activated = True

        pnl = (c - entry) if direction == 'BUY' else (entry - c)
        return 'EOD', c, future_data.index[-1], pnl

    def _build_results(self):
        if not self.trades:
            return {
                'trades': pd.DataFrame(),
                'metrics': self._empty_metrics(),
            }

        df = pd.DataFrame(self.trades)
        df['cum_pnl_pts'] = df['pnl_pts'].cumsum()
        df['cum_pnl_dollars'] = df['pnl_dollars'].cumsum()
        df['peak'] = df['cum_pnl_pts'].cummax()
        df['drawdown'] = df['cum_pnl_pts'] - df['peak']
        df['trade_date'] = pd.to_datetime(df['entry_time']).dt.date

        metrics = self._compute_metrics(df)

        return {
            'trades': df,
            'metrics': metrics,
        }

    def _compute_metrics(self, df):
        total = len(df)
        wins = len(df[df['result'] == 'WIN'])
        losses = len(df[df['result'] == 'LOSS'])
        bes = len(df[df['result'] == 'BE'])
        eods = len(df[df['result'] == 'EOD'])

        win_rate = (wins / total * 100) if total > 0 else 0
        total_pnl = df['pnl_pts'].sum()
        total_pnl_dollars = df['pnl_dollars'].sum()
        avg_win = df[df['pnl_pts'] > 0]['pnl_pts'].mean() if wins > 0 else 0
        avg_loss = df[df['pnl_pts'] < 0]['pnl_pts'].mean() if losses > 0 else 0
        profit_factor = abs(df[df['pnl_pts'] > 0]['pnl_pts'].sum() / df[df['pnl_pts'] < 0]['pnl_pts'].sum()) if df[df['pnl_pts'] < 0]['pnl_pts'].sum() != 0 else float('inf')

        max_dd = df['drawdown'].min() if 'drawdown' in df.columns else 0
        max_dd_dollars = max_dd * self.config['contract_value']

        avg_rr = df[df['result'] == 'WIN']['rr_achieved'].mean() if wins > 0 else 0

        daily = df.groupby('trade_date')['pnl_pts'].sum()
        best_day = daily.max() if len(daily) > 0 else 0
        worst_day = daily.min() if len(daily) > 0 else 0
        avg_daily = daily.mean() if len(daily) > 0 else 0
        winning_days = (daily > 0).sum()
        losing_days = (daily < 0).sum()
        total_days = len(daily)

        consecutive_wins = 0
        consecutive_losses = 0
        max_cons_wins = 0
        max_cons_losses = 0
        for pnl in df['pnl_pts']:
            if pnl > 0:
                consecutive_wins += 1
                consecutive_losses = 0
                max_cons_wins = max(max_cons_wins, consecutive_wins)
            elif pnl < 0:
                consecutive_losses += 1
                consecutive_wins = 0
                max_cons_losses = max(max_cons_losses, consecutive_losses)
            else:
                consecutive_wins = 0
                consecutive_losses = 0

        trades_per_month = total / max(total_days / 21, 1)

        return {
            'total_trades': total,
            'wins': wins,
            'losses': losses,
            'breakevens': bes,
            'eod_exits': eods,
            'win_rate': round(win_rate, 1),
            'total_pnl_pts': round(total_pnl, 2),
            'total_pnl_dollars': round(total_pnl_dollars, 2),
            'avg_win_pts': round(avg_win, 2),
            'avg_loss_pts': round(avg_loss, 2),
            'profit_factor': round(profit_factor, 2),
            'max_drawdown_pts': round(max_dd, 2),
            'max_drawdown_dollars': round(max_dd_dollars, 2),
            'avg_rr_on_wins': round(avg_rr, 2),
            'best_day_pts': round(best_day, 2),
            'worst_day_pts': round(worst_day, 2),
            'avg_daily_pnl': round(avg_daily, 2),
            'winning_days': winning_days,
            'losing_days': losing_days,
            'total_trading_days': total_days,
            'max_consecutive_wins': max_cons_wins,
            'max_consecutive_losses': max_cons_losses,
            'trades_per_month': round(trades_per_month, 1),
        }

    def _empty_metrics(self):
        keys = [
            'total_trades', 'wins', 'losses', 'breakevens', 'eod_exits',
            'win_rate', 'total_pnl_pts', 'total_pnl_dollars',
            'avg_win_pts', 'avg_loss_pts', 'profit_factor',
            'max_drawdown_pts', 'max_drawdown_dollars', 'avg_rr_on_wins',
            'best_day_pts', 'worst_day_pts', 'avg_daily_pnl',
            'winning_days', 'losing_days', 'total_trading_days',
            'max_consecutive_wins', 'max_consecutive_losses',
            'trades_per_month',
        ]
        return {k: 0 for k in keys}
