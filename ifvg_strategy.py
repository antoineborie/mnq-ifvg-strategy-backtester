import pandas as pd
import numpy as np


class IFVGStrategy:
    def __init__(self, config=None):
        defaults = {
            'min_fvg_size': 4.0,
            'max_fvg_age': 30,
            'rr_target': 2.5,
            'max_risk_pts': 50.0,
            'min_risk_pts': 5.0,
            'max_trades_per_day': 3,
            'killzone_start': '09:30',
            'killzone_end': '11:00',
            'use_be': True,
            'be_trigger_rr': 1.5,
            'cooldown_minutes': 5,
            'displacement_min': 3.0,
            'contract_value': 2.0,
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
            self._process_day(day_data, day)

        return self._build_results()

    def _process_day(self, day_data, day):
        ks = self.config['killzone_start']
        ke = self.config['killzone_end']

        killzone = day_data.between_time(ks, ke)
        if len(killzone) < 5:
            return

        pre_kz = day_data[day_data.index < killzone.index[0]]
        if len(pre_kz) < 10:
            return

        fvgs = self._detect_fvgs(pre_kz)

        trades_today = 0
        max_trades = self.config['max_trades_per_day']
        last_trade_time = None
        cooldown = self.config['cooldown_minutes']

        for i in range(1, len(killzone)):
            if trades_today >= max_trades:
                break

            current_time = killzone.index[i]
            current_bar = killzone.iloc[i]
            prev_bar = killzone.iloc[i - 1]

            if last_trade_time is not None:
                minutes_since = (current_time - last_trade_time).total_seconds() / 60
                if minutes_since < cooldown:
                    continue

            new_fvgs = self._detect_fvgs_in_range(killzone.iloc[:i])
            all_fvgs = fvgs + new_fvgs

            seen_ids = set()
            unique_fvgs = []
            for f in all_fvgs:
                if f['id'] not in seen_ids and not f['filled']:
                    seen_ids.add(f['id'])
                    unique_fvgs.append(f)

            ifvg_signals = self._check_ifvg(unique_fvgs, current_bar, prev_bar, current_time)

            for signal in ifvg_signals:
                if trades_today >= max_trades:
                    break

                remaining = day_data[day_data.index > current_time]
                trade = self._execute_trade(signal, current_bar, current_time, remaining)
                if trade:
                    self.trades.append(trade)
                    trades_today += 1
                    last_trade_time = current_time
                    for f in all_fvgs:
                        if f['id'] == signal['fvg_id']:
                            f['filled'] = True
                    break

    def _detect_fvgs(self, data):
        return self._scan_fvgs(data, prefix="pre")

    def _detect_fvgs_in_range(self, data):
        return self._scan_fvgs(data, prefix="kz")

    def _scan_fvgs(self, data, prefix=""):
        fvgs = []
        if len(data) < 3:
            return fvgs

        highs = data['high'].values
        lows = data['low'].values
        closes = data['close'].values
        opens = data['open'].values
        times = data.index
        min_size = self.config['min_fvg_size']

        for i in range(2, len(data)):
            gap_up = lows[i] - highs[i - 2]
            if gap_up >= min_size:
                mid_body = closes[i - 1] - opens[i - 1]
                if mid_body > 0:
                    fvgs.append({
                        'id': f"{prefix}_bull_{times[i]}",
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
                        'id': f"{prefix}_bear_{times[i]}",
                        'type': 'bearish',
                        'top': float(lows[i - 2]),
                        'bottom': float(highs[i]),
                        'midpoint': float((lows[i - 2] + highs[i]) / 2),
                        'size': float(gap_down),
                        'time': times[i],
                        'filled': False,
                    })

        return fvgs

    def _check_ifvg(self, fvgs, bar, prev_bar, current_time):
        signals = []
        max_age = self.config['max_fvg_age']
        disp_min = self.config['displacement_min']
        high = bar['high']
        low = bar['low']
        close = bar['close']
        open_ = bar['open']

        for fvg in fvgs:
            if fvg['filled']:
                continue

            age_minutes = (current_time - fvg['time']).total_seconds() / 60
            if age_minutes > max_age or age_minutes < 1:
                continue

            if fvg['type'] == 'bullish':
                if low <= fvg['bottom']:
                    displacement = open_ - close
                    if displacement < disp_min:
                        continue

                    signals.append({
                        'direction': 'SELL',
                        'fvg_id': fvg['id'],
                        'fvg_type': fvg['type'],
                        'fvg_top': fvg['top'],
                        'fvg_bottom': fvg['bottom'],
                        'fvg_midpoint': fvg['midpoint'],
                        'fvg_size': fvg['size'],
                        'fvg_age': age_minutes,
                    })

            elif fvg['type'] == 'bearish':
                if high >= fvg['top']:
                    displacement = close - open_
                    if displacement < disp_min:
                        continue

                    signals.append({
                        'direction': 'BUY',
                        'fvg_id': fvg['id'],
                        'fvg_type': fvg['type'],
                        'fvg_top': fvg['top'],
                        'fvg_bottom': fvg['bottom'],
                        'fvg_midpoint': fvg['midpoint'],
                        'fvg_size': fvg['size'],
                        'fvg_age': age_minutes,
                    })

        return signals

    def _execute_trade(self, signal, entry_bar, entry_time, future_data):
        if future_data.empty:
            return None

        direction = signal['direction']
        rr = self.config['rr_target']
        max_risk = self.config['max_risk_pts']
        min_risk = self.config['min_risk_pts']
        use_be = self.config['use_be']
        be_trigger = self.config['be_trigger_rr']
        contract_val = self.config['contract_value']

        if direction == 'SELL':
            entry_price = entry_bar['close']
            sl_price = signal['fvg_top'] + 2.0
            risk = sl_price - entry_price
            if risk <= 0 or risk > max_risk or risk < min_risk:
                return None
            tp_price = entry_price - (risk * rr)
        else:
            entry_price = entry_bar['close']
            sl_price = signal['fvg_bottom'] - 2.0
            risk = entry_price - sl_price
            if risk <= 0 or risk > max_risk or risk < min_risk:
                return None
            tp_price = entry_price + (risk * rr)

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
            'fvg_type': signal['fvg_type'],
            'fvg_size': round(signal['fvg_size'], 2),
            'fvg_age_min': round(signal['fvg_age'], 1),
        }

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
        }

    def _empty_metrics(self):
        return {k: 0 for k in [
            'total_trades', 'wins', 'losses', 'breakevens', 'eod_exits',
            'win_rate', 'total_pnl_pts', 'total_pnl_dollars',
            'avg_win_pts', 'avg_loss_pts', 'profit_factor',
            'max_drawdown_pts', 'max_drawdown_dollars', 'avg_rr_on_wins',
            'best_day_pts', 'worst_day_pts', 'avg_daily_pnl',
            'winning_days', 'losing_days', 'total_trading_days',
            'max_consecutive_wins', 'max_consecutive_losses',
        ]}
