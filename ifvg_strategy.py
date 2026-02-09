import pandas as pd
import numpy as np


class IFVGStrategy:
    def __init__(self, config=None):
        defaults = {
            'min_fvg_size': 3.0,
            'max_fvg_age_m15': 15,
            'rr_target': 0.8,
            'max_risk_pts': 25.0,
            'min_risk_pts': 5.0,
            'max_trades_per_day': 1,
            'killzone_start': '09:30',
            'killzone_end': '12:00',
            'use_be': True,
            'be_trigger_rr': 0.5,
            'cooldown_minutes': 10,
            'contract_value': 2.0,
            'target_mode': 'fixed_rr',
            'retracement_pct': 60,
            'structure_lookback_days': 20,
            'use_displacement_filter': True,
            'min_displacement_body_pct': 55,
            'min_displacement_size': 3.5,
            'use_liquidity_sweep': False,
            'sweep_lookback_bars': 20,
            'use_m1_confirmation': True,
            'use_structure_confluence': False,
            'confluence_distance_pts': 50.0,
            'use_trend_filter': False,
            'trend_lookback_days': 3,
            'use_day_filter': False,
            'allowed_days': [0, 1, 2, 3, 4],
            'use_session_momentum': False,
            'momentum_threshold': 0.4,
            'use_range_filter': False,
            'min_prev_day_range': 60.0,
            'max_prev_day_range': 400.0,
            'use_trailing_stop': True,
            'trail_trigger_rr': 0.3,
            'trail_offset_pct': 25,
            'entry_start_time': '10:00',
            'use_m1_momentum': False,
            'momentum_bars': 5,
            'momentum_min_score': 3,
            'use_volatility_regime': False,
            'vol_atr_period': 10,
            'vol_low_percentile': 30,
            'vol_high_percentile': 70,
            'vol_low_rr_target': 1.0,
            'vol_low_min_fvg_mult': 0.8,
            'vol_low_max_trades': 1,
            'vol_low_min_displacement_mult': 1.2,
            'vol_high_max_risk_mult': 0.8,
            'use_enhanced_bias': False,
            'bias_opening_range_minutes': 15,
            'bias_prev_day_weight': True,
            'bias_min_confidence': 2,
            'use_stop_after_loss': True,
            'use_fvg_freshness': False,
            'max_fvg_age_minutes': 120,
            'use_multi_confirm': False,
            'confirm_require_wick_rejection': True,
            'confirm_require_body_ratio': 40,
            'use_opening_range_filter': False,
            'opening_range_bias_only': True,
            'partial_tp_pct': 100,
        }
        self.config = {**defaults, **(config or {})}
        self.trades = []

    def run_backtest(self, df_utc):
        self.trades = []

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
            day_open=('open', 'first'),
            day_close=('close', 'last'),
        )

        daily_ohlc['day_range'] = daily_ohlc['day_high'] - daily_ohlc['day_low']

        atr_period = self.config['vol_atr_period']
        all_ranges = daily_ohlc['day_range'].values
        vol_low_pct = np.percentile(all_ranges, self.config['vol_low_percentile']) if len(all_ranges) > atr_period else 0
        vol_high_pct = np.percentile(all_ranges, self.config['vol_high_percentile']) if len(all_ranges) > atr_period else 999

        lookback_days = self.config['structure_lookback_days']

        for i, day in enumerate(trading_days):
            day_data = grouped[day]
            if len(day_data) < 30:
                continue

            if self.config['use_day_filter']:
                import datetime
                weekday = day.weekday() if hasattr(day, 'weekday') else datetime.date(day.year, day.month, day.day).weekday()
                if weekday not in self.config['allowed_days']:
                    continue

            start_idx = max(0, i - lookback_days)
            history_days = trading_days[start_idx:i]
            history_ohlc = daily_ohlc.loc[daily_ohlc.index.isin(history_days)] if history_days else pd.DataFrame()

            if self.config['use_range_filter'] and len(history_ohlc) > 0:
                prev_day = history_ohlc.iloc[-1]
                prev_range = prev_day['day_high'] - prev_day['day_low']
                if prev_range < self.config['min_prev_day_range'] or prev_range > self.config['max_prev_day_range']:
                    continue

            if self.config['use_trend_filter'] and len(history_ohlc) >= self.config['trend_lookback_days']:
                trend_bias = self._get_multi_day_trend(history_ohlc)
            else:
                trend_bias = None

            vol_regime = 'normal'
            if self.config['use_volatility_regime'] and len(history_ohlc) >= atr_period:
                recent_atr = float(history_ohlc['day_range'].tail(atr_period).mean())
                if recent_atr <= vol_low_pct:
                    vol_regime = 'low'
                elif recent_atr >= vol_high_pct:
                    vol_regime = 'high'

            self._process_day(day_data, history_ohlc, day, df, trend_bias, vol_regime)

        return self._build_results()

    def _get_multi_day_trend(self, history_ohlc):
        n = self.config['trend_lookback_days']
        recent = history_ohlc.tail(n)
        if len(recent) < 2:
            return None

        closes = recent['day_close'].values
        opens = recent['day_open'].values
        highs = recent['day_high'].values
        lows = recent['day_low'].values

        bullish_days = sum(1 for i in range(len(closes)) if closes[i] > opens[i])
        bearish_days = sum(1 for i in range(len(closes)) if closes[i] < opens[i])

        higher_highs = sum(1 for i in range(1, len(highs)) if highs[i] > highs[i-1])
        lower_lows = sum(1 for i in range(1, len(lows)) if lows[i] < lows[i-1])

        overall_move = closes[-1] - opens[0]

        bull_score = bullish_days + higher_highs + (1 if overall_move > 0 else 0)
        bear_score = bearish_days + lower_lows + (1 if overall_move < 0 else 0)

        if bull_score >= bear_score + 2:
            return 'BUY'
        elif bear_score >= bull_score + 2:
            return 'SELL'
        return None

    def _get_regime_config(self, vol_regime):
        cfg = dict(self.config)
        if vol_regime == 'low':
            cfg['rr_target'] = self.config['vol_low_rr_target']
            cfg['min_fvg_size'] = self.config['min_fvg_size'] * self.config['vol_low_min_fvg_mult']
            cfg['max_trades_per_day'] = self.config['vol_low_max_trades']
            cfg['min_displacement_body_pct'] = min(
                80, self.config['min_displacement_body_pct'] * self.config['vol_low_min_displacement_mult']
            )
            cfg['min_displacement_size'] = self.config['min_displacement_size'] * self.config['vol_low_min_fvg_mult']
        elif vol_regime == 'high':
            cfg['max_risk_pts'] = self.config['max_risk_pts'] * self.config['vol_high_max_risk_mult']
        return cfg

    def _process_day(self, day_data, history_ohlc, day, full_df, trend_bias, vol_regime='normal'):
        if self.config['use_volatility_regime'] and vol_regime != 'normal':
            active_cfg = self._get_regime_config(vol_regime)
        else:
            active_cfg = self.config

        ks = active_cfg['killzone_start']
        ke = active_cfg['killzone_end']

        killzone_m1 = day_data.between_time(ks, ke)
        if len(killzone_m1) < 5:
            return

        structure = self._compute_structure_levels_fast(history_ohlc, day_data)

        if active_cfg.get('use_enhanced_bias', False):
            h1_bias, bias_confidence = self._determine_enhanced_bias(day_data, killzone_m1, history_ohlc)
            if h1_bias is None:
                return
            min_conf = active_cfg.get('bias_min_confidence', 2)
            if bias_confidence < min_conf:
                return
        else:
            h1_bias = self._determine_h1_bias(day_data, killzone_m1)
            if h1_bias is None:
                return
            bias_confidence = 1

        if active_cfg['use_trend_filter'] and trend_bias is not None:
            if trend_bias != h1_bias:
                return

        if active_cfg['use_session_momentum']:
            momentum_ok = self._check_session_momentum(day_data, killzone_m1, h1_bias)
            if not momentum_ok:
                return

        if active_cfg.get('use_opening_range_filter', False):
            or_data = day_data.between_time('09:30', '09:45')
            if len(or_data) >= 5:
                or_open = or_data.iloc[0]['open']
                or_close = or_data.iloc[-1]['close']
                or_high = or_data['high'].max()
                or_low = or_data['low'].min()
                or_range = or_high - or_low
                or_body = or_close - or_open
                if or_range > 0 and abs(or_body) / or_range > 0.3:
                    or_bias = 'BUY' if or_body > 0 else 'SELL'
                    if or_bias != h1_bias:
                        return

        pre_kz = day_data[day_data.index < killzone_m1.index[0]]
        kz_and_before = pd.concat([pre_kz, killzone_m1]).sort_index()

        m15_data = kz_and_before.resample('15min').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna()

        if len(m15_data) < 4:
            return

        m15_fvgs = self._detect_fvgs_m15(m15_data, min_size_override=active_cfg['min_fvg_size'])

        if active_cfg['use_displacement_filter']:
            m15_fvgs = self._filter_by_displacement(m15_fvgs, m15_data, active_cfg)

        m15_ifvgs = self._detect_ifvgs_m15(m15_fvgs, m15_data, h1_bias, structure)

        if active_cfg.get('use_fvg_freshness', False):
            max_age_min = active_cfg.get('max_fvg_age_minutes', 120)
            m15_ifvgs = self._filter_by_freshness(m15_ifvgs, killzone_m1, max_age_min)

        if active_cfg['use_structure_confluence']:
            m15_ifvgs = self._filter_by_confluence(m15_ifvgs, structure)

        if active_cfg['use_liquidity_sweep']:
            m15_ifvgs = self._filter_by_liquidity_sweep(m15_ifvgs, m15_data, killzone_m1)

        trades_today = 0
        max_trades = active_cfg['max_trades_per_day']
        last_trade_time = None
        cooldown = active_cfg['cooldown_minutes']
        used_ifvgs = set()
        had_loss_today = False

        kz_highs = killzone_m1['high'].values
        kz_lows = killzone_m1['low'].values
        kz_closes = killzone_m1['close'].values
        kz_opens = killzone_m1['open'].values
        kz_times = killzone_m1.index

        entry_start = active_cfg.get('entry_start_time', '09:30')
        entry_start_parts = entry_start.split(':')
        entry_start_minutes = int(entry_start_parts[0]) * 60 + int(entry_start_parts[1])

        for i in range(len(killzone_m1)):
            if trades_today >= max_trades:
                break

            if active_cfg.get('use_stop_after_loss', False) and had_loss_today:
                break

            current_time = kz_times[i]

            current_minutes = current_time.hour * 60 + current_time.minute
            if current_minutes < entry_start_minutes:
                continue

            if last_trade_time is not None:
                minutes_since = (current_time - last_trade_time).total_seconds() / 60
                if minutes_since < cooldown:
                    continue

            current_bar = {
                'high': kz_highs[i],
                'low': kz_lows[i],
                'close': kz_closes[i],
                'open': kz_opens[i],
            }

            for ifvg in m15_ifvgs:
                if ifvg['id'] in used_ifvgs:
                    continue
                if trades_today >= max_trades:
                    break
                if ifvg['direction'] != h1_bias:
                    continue

                entry_signal = self._check_m1_retracement(ifvg, current_bar, current_time)
                if entry_signal:
                    if active_cfg.get('use_multi_confirm', False):
                        if not self._check_enhanced_confirmation(killzone_m1, i, ifvg['direction'], active_cfg):
                            continue
                    elif active_cfg['use_m1_confirmation']:
                        if not self._check_m1_confirmation(current_bar, ifvg['direction']):
                            continue

                    if active_cfg.get('use_m1_momentum', False):
                        if not self._check_m1_momentum(killzone_m1, i, ifvg['direction']):
                            continue

                    remaining = day_data[day_data.index > current_time]
                    trade = self._execute_trade(
                        entry_signal, current_bar, current_time, remaining, structure, active_cfg
                    )
                    if trade:
                        trade['vol_regime'] = vol_regime
                        trade['bias_confidence'] = bias_confidence
                        self.trades.append(trade)
                        trades_today += 1
                        last_trade_time = current_time
                        used_ifvgs.add(ifvg['id'])
                        if trade['result'] == 'LOSS':
                            had_loss_today = True
                        break

    def _check_session_momentum(self, day_data, killzone, h1_bias):
        kz_start = killzone.index[0]
        pre_session = day_data.between_time('08:00', '09:30')
        if len(pre_session) < 10:
            return True

        opens = pre_session['open'].values
        closes = pre_session['close'].values
        highs = pre_session['high'].values
        lows = pre_session['low'].values

        session_range = highs.max() - lows.min()
        if session_range == 0:
            return True

        net_move = closes[-1] - opens[0]
        momentum = net_move / session_range

        threshold = self.config['momentum_threshold']

        if h1_bias == 'BUY' and momentum < -threshold:
            return False
        if h1_bias == 'SELL' and momentum > threshold:
            return False

        return True

    def _filter_by_displacement(self, fvgs, m15_data, cfg_override=None):
        cfg = cfg_override or self.config
        filtered = []
        min_body_pct = cfg['min_displacement_body_pct'] / 100.0
        min_disp_size = cfg['min_displacement_size']

        opens = m15_data['open'].values
        closes = m15_data['close'].values
        highs = m15_data['high'].values
        lows = m15_data['low'].values

        for fvg in fvgs:
            idx_str = fvg['id'].split('_')
            try:
                bar_idx = int(idx_str[2])
            except (IndexError, ValueError):
                filtered.append(fvg)
                continue

            if bar_idx - 1 < 0 or bar_idx - 1 >= len(opens):
                filtered.append(fvg)
                continue

            mid_idx = bar_idx - 1
            body = abs(closes[mid_idx] - opens[mid_idx])
            total_range = highs[mid_idx] - lows[mid_idx]

            if total_range == 0:
                continue

            body_pct = body / total_range

            if body_pct >= min_body_pct and body >= min_disp_size:
                filtered.append(fvg)

        return filtered

    def _check_m1_momentum(self, killzone_m1, current_idx, direction):
        n_bars = self.config.get('momentum_bars', 5)
        min_score = self.config.get('momentum_min_score', 3)

        start_idx = max(0, current_idx - n_bars)
        if start_idx == current_idx:
            return True

        closes = killzone_m1['close'].values[start_idx:current_idx]
        opens = killzone_m1['open'].values[start_idx:current_idx]

        if len(closes) < 2:
            return True

        score = 0
        for j in range(len(closes)):
            if direction == 'SELL' and closes[j] < opens[j]:
                score += 1
            elif direction == 'BUY' and closes[j] > opens[j]:
                score += 1

        net_move = closes[-1] - closes[0]
        if direction == 'SELL' and net_move < 0:
            score += 1
        elif direction == 'BUY' and net_move > 0:
            score += 1

        return score >= min_score

    def _filter_by_confluence(self, ifvgs, structure):
        filtered = []
        max_dist = self.config['confluence_distance_pts']

        all_levels = []
        for key in ['daily_highs', 'daily_lows', 'swing_highs', 'swing_lows', 'weekly_highs', 'weekly_lows']:
            all_levels.extend(structure.get(key, []))

        if structure.get('pdh'):
            all_levels.append(structure['pdh'])
        if structure.get('pdl'):
            all_levels.append(structure['pdl'])

        if not all_levels:
            return ifvgs

        for ifvg in ifvgs:
            zone_mid = (ifvg['zone_top'] + ifvg['zone_bottom']) / 2

            near_level = any(abs(zone_mid - lvl) <= max_dist for lvl in all_levels)
            if near_level:
                filtered.append(ifvg)

        return filtered if filtered else ifvgs[:1]

    def _filter_by_liquidity_sweep(self, ifvgs, m15_data, killzone):
        filtered = []
        lookback = self.config['sweep_lookback_bars']

        for ifvg in ifvgs:
            inv_time = ifvg['inversion_time']
            pre_inv = m15_data[m15_data.index <= inv_time].tail(lookback)

            if len(pre_inv) < 3:
                filtered.append(ifvg)
                continue

            recent_high = pre_inv['high'].max()
            recent_low = pre_inv['low'].min()

            post_inv = m15_data[m15_data.index >= inv_time].head(3)
            if post_inv.empty:
                continue

            if ifvg['direction'] == 'SELL':
                if post_inv['high'].max() >= recent_high * 0.999:
                    filtered.append(ifvg)
            elif ifvg['direction'] == 'BUY':
                if post_inv['low'].min() <= recent_low * 1.001:
                    filtered.append(ifvg)

        return filtered

    def _filter_by_freshness(self, ifvgs, killzone, max_age_minutes):
        if not ifvgs or killzone.empty:
            return ifvgs

        filtered = []
        kz_start = killzone.index[0]

        for ifvg in ifvgs:
            inv_time = ifvg.get('inversion_time')
            if inv_time is None:
                filtered.append(ifvg)
                continue
            age_minutes = (kz_start - inv_time).total_seconds() / 60
            if age_minutes <= max_age_minutes:
                filtered.append(ifvg)

        return filtered if filtered else ifvgs[:1]

    def _check_enhanced_confirmation(self, killzone_m1, bar_idx, direction, cfg):
        if bar_idx < 1 or bar_idx >= len(killzone_m1):
            return False

        current = killzone_m1.iloc[bar_idx]
        body = current['close'] - current['open']
        total_range = current['high'] - current['low']

        if total_range == 0:
            return False

        min_body_ratio = cfg.get('confirm_require_body_ratio', 40) / 100.0
        require_wick = cfg.get('confirm_require_wick_rejection', True)

        if direction == 'SELL':
            if body >= 0:
                return False
            if abs(body) / total_range < min_body_ratio:
                return False
            if require_wick:
                upper_wick = current['high'] - max(current['open'], current['close'])
                if upper_wick < abs(body) * 0.2:
                    if abs(body) / total_range < 0.6:
                        return False
            prev = killzone_m1.iloc[bar_idx - 1]
            if current['close'] > prev['close']:
                return False
            return True

        elif direction == 'BUY':
            if body <= 0:
                return False
            if body / total_range < min_body_ratio:
                return False
            if require_wick:
                lower_wick = min(current['open'], current['close']) - current['low']
                if lower_wick < body * 0.2:
                    if body / total_range < 0.6:
                        return False
            prev = killzone_m1.iloc[bar_idx - 1]
            if current['close'] < prev['close']:
                return False
            return True

        return False

    def _check_m1_confirmation(self, bar, direction):
        body = bar['close'] - bar['open']
        total_range = bar['high'] - bar['low']

        if total_range == 0:
            return False

        if direction == 'SELL':
            if body < 0 and abs(body) / total_range > 0.3:
                upper_wick = bar['high'] - max(bar['open'], bar['close'])
                if upper_wick > abs(body) * 0.3:
                    return True
                if abs(body) / total_range > 0.5:
                    return True
            return False

        elif direction == 'BUY':
            if body > 0 and body / total_range > 0.3:
                lower_wick = min(bar['open'], bar['close']) - bar['low']
                if lower_wick > body * 0.3:
                    return True
                if body / total_range > 0.5:
                    return True
            return False

        return False

    def _compute_structure_levels_fast(self, history_ohlc, day_data):
        levels = {
            'daily_highs': [],
            'daily_lows': [],
            'weekly_highs': [],
            'weekly_lows': [],
            'swing_highs': [],
            'swing_lows': [],
        }

        if history_ohlc.empty:
            return levels

        highs = history_ohlc['day_high'].values
        lows = history_ohlc['day_low'].values

        levels['daily_highs'] = [float(h) for h in highs]
        levels['daily_lows'] = [float(l) for l in lows]

        n = len(highs)
        week_size = 5
        for ws in range(0, n, week_size):
            we = min(ws + week_size, n)
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

        if levels['daily_highs']:
            levels['pdh'] = levels['daily_highs'][0]
        if levels['daily_lows']:
            levels['pdl'] = levels['daily_lows'][0]

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
        return 'BUY' if last_h1['close'] > last_h1['open'] else 'SELL'

    def _determine_enhanced_bias(self, day_data, killzone, history_ohlc):
        if killzone.empty:
            return None, 0

        kz_start = killzone.index[0]
        confidence = 0

        pre_kz = day_data[day_data.index < kz_start]
        if len(pre_kz) < 10:
            session_start = day_data.between_time('08:30', '09:30')
            if session_start.empty:
                return None, 0
            pre_kz = session_start

        h1 = pre_kz.resample('1h').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
        }).dropna()

        h1_bias = None
        if not h1.empty:
            last_h1 = h1.iloc[-1]
            h1_body = last_h1['close'] - last_h1['open']
            h1_range = last_h1['high'] - last_h1['low']
            if h1_range > 0 and abs(h1_body) / h1_range > 0.3:
                h1_bias = 'BUY' if h1_body > 0 else 'SELL'
                confidence += 1
                if abs(h1_body) / h1_range > 0.6:
                    confidence += 1
        elif not pre_kz.empty:
            o = pre_kz.iloc[0]['open']
            c = pre_kz.iloc[-1]['close']
            h1_bias = 'BUY' if c > o else 'SELL'
            confidence += 1

        if h1_bias is None:
            return None, 0

        or_minutes = self.config.get('bias_opening_range_minutes', 15)
        or_data = day_data.between_time('09:30', f'09:{30 + or_minutes}')
        if len(or_data) >= 5:
            or_open = or_data.iloc[0]['open']
            or_close = or_data.iloc[-1]['close']
            or_bias = 'BUY' if or_close > or_open else 'SELL'
            if or_bias == h1_bias:
                confidence += 1

        if self.config.get('bias_prev_day_weight', True) and len(history_ohlc) > 0:
            prev = history_ohlc.iloc[-1]
            prev_close = prev['day_close']
            prev_open = prev['day_open']
            prev_bias = 'BUY' if prev_close > prev_open else 'SELL'
            if prev_bias == h1_bias:
                confidence += 1

        pre_market = day_data.between_time('04:00', '09:30')
        if len(pre_market) >= 30:
            pm_open = pre_market.iloc[0]['open']
            pm_close = pre_market.iloc[-1]['close']
            pm_bias = 'BUY' if pm_close > pm_open else 'SELL'
            if pm_bias == h1_bias:
                confidence += 1

        return h1_bias, confidence

    def _detect_fvgs_m15(self, m15_data, min_size_override=None):
        fvgs = []
        if len(m15_data) < 3:
            return fvgs

        highs = m15_data['high'].values
        lows = m15_data['low'].values
        closes = m15_data['close'].values
        opens = m15_data['open'].values
        times = m15_data.index
        min_size = min_size_override if min_size_override is not None else self.config['min_fvg_size']

        for i in range(2, len(m15_data)):
            gap_up = lows[i] - highs[i - 2]
            if gap_up >= min_size:
                mid_body = closes[i - 1] - opens[i - 1]
                if mid_body > 0:
                    fvgs.append({
                        'id': f"m15_bull_{i}_{times[i]}",
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
                        'id': f"m15_bear_{i}_{times[i]}",
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

            if len(later_bars) > max_age:
                later_bars = later_bars.iloc[:max_age]

            closes = later_bars['close'].values
            times = later_bars.index

            if fvg['type'] == 'bullish' and h1_bias == 'SELL':
                for j in range(len(closes)):
                    if closes[j] < fvg['bottom']:
                        ifvgs.append({
                            'id': fvg['id'],
                            'direction': 'SELL',
                            'fvg_top': fvg['top'],
                            'fvg_bottom': fvg['bottom'],
                            'fvg_midpoint': fvg['midpoint'],
                            'fvg_size': fvg['size'],
                            'inversion_time': times[j],
                            'zone_top': fvg['top'],
                            'zone_bottom': fvg['midpoint'],
                        })
                        fvg['filled'] = True
                        break

            elif fvg['type'] == 'bearish' and h1_bias == 'BUY':
                for j in range(len(closes)):
                    if closes[j] > fvg['top']:
                        ifvgs.append({
                            'id': fvg['id'],
                            'direction': 'BUY',
                            'fvg_top': fvg['top'],
                            'fvg_bottom': fvg['bottom'],
                            'fvg_midpoint': fvg['fvg_midpoint'] if 'fvg_midpoint' in fvg else fvg['midpoint'],
                            'fvg_size': fvg['size'],
                            'inversion_time': times[j],
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

        if ifvg['direction'] == 'SELL':
            entry_level_bottom = zone_top - (zone_range * ret_pct)
            if bar['high'] >= entry_level_bottom and bar['close'] < zone_top:
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
            adjusted_top = zone_bottom + (zone_range * ret_pct)
            if bar['low'] <= adjusted_top and bar['close'] > zone_bottom:
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

    def _execute_trade(self, signal, entry_bar, entry_time, future_data, structure, cfg_override=None):
        if future_data.empty:
            return None

        cfg = cfg_override or self.config
        direction = signal['direction']
        rr = cfg['rr_target']
        max_risk = cfg['max_risk_pts']
        min_risk = cfg['min_risk_pts']
        use_be = cfg['use_be']
        be_trigger = cfg['be_trigger_rr']
        contract_val = cfg['contract_value']
        target_mode = cfg['target_mode']
        use_trail = cfg.get('use_trailing_stop', False)
        trail_trigger = cfg.get('trail_trigger_rr', 1.0)
        trail_offset_pct = cfg.get('trail_offset_pct', 50) / 100.0
        partial_tp_pct = cfg.get('partial_tp_pct', 100)

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
            future_data, entry_price, tp_price, sl_price, direction, risk,
            use_be, be_trigger, use_trail, trail_trigger, trail_offset_pct,
            partial_tp_pct
        )

        pnl_dollars = pnl_pts * contract_val

        tp_full_distance = abs(tp_price - entry_price)
        partial_tp_level = entry_price + (tp_full_distance * partial_tp_pct / 100.0) if direction == 'BUY' else entry_price - (tp_full_distance * partial_tp_pct / 100.0)
        tp_pct_reached = round(abs(exit_price - entry_price) / tp_full_distance * 100, 1) if tp_full_distance > 0 else 0

        return {
            'entry_time': entry_time,
            'exit_time': exit_time,
            'direction': direction,
            'entry': round(entry_price, 2),
            'sl': round(sl_price, 2),
            'tp': round(tp_price, 2),
            'tp_partial': round(partial_tp_level, 2),
            'exit_price': round(exit_price, 2),
            'risk_pts': round(risk, 2),
            'pnl_pts': round(pnl_pts, 2),
            'pnl_dollars': round(pnl_dollars, 2),
            'result': result,
            'rr_achieved': round(pnl_pts / risk, 2) if risk > 0 else 0,
            'tp_pct_reached': tp_pct_reached,
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

    def _simulate_trade(self, future_data, entry, tp, sl, direction, risk,
                        use_be, be_trigger, use_trail=False, trail_trigger=1.0, trail_offset_pct=0.5,
                        partial_tp_pct=100):
        current_sl = sl
        be_activated = False
        trail_activated = False
        best_price = entry

        highs = future_data['high'].values
        lows = future_data['low'].values
        closes = future_data['close'].values
        times = future_data.index

        be_level_buy = entry + (risk * be_trigger) if use_be else None
        be_level_sell = entry - (risk * be_trigger) if use_be else None

        trail_level_buy = entry + (risk * trail_trigger) if use_trail else None
        trail_level_sell = entry - (risk * trail_trigger) if use_trail else None

        for i in range(len(future_data)):
            h, l, c = highs[i], lows[i], closes[i]

            if direction == 'BUY':
                if h > best_price:
                    best_price = h

                if l <= current_sl:
                    pnl = current_sl - entry
                    if pnl > 0:
                        label = 'BE'
                    else:
                        label = 'LOSS'
                    return label, current_sl, times[i], pnl
                if h >= tp:
                    return 'WIN', tp, times[i], tp - entry

                if use_trail and trail_level_buy is not None and h >= trail_level_buy:
                    trail_activated = True
                if trail_activated:
                    trail_sl = best_price - (risk * trail_offset_pct)
                    if trail_sl > current_sl:
                        current_sl = trail_sl

                if use_be and not be_activated and not trail_activated and be_level_buy is not None and h >= be_level_buy:
                    current_sl = entry + 1.0
                    be_activated = True

            else:
                if l < best_price:
                    best_price = l

                if h >= current_sl:
                    pnl = entry - current_sl
                    if pnl > 0:
                        label = 'BE'
                    else:
                        label = 'LOSS'
                    return label, current_sl, times[i], pnl
                if l <= tp:
                    return 'WIN', tp, times[i], entry - tp

                if use_trail and trail_level_sell is not None and l <= trail_level_sell:
                    trail_activated = True
                if trail_activated:
                    trail_sl = best_price + (risk * trail_offset_pct)
                    if trail_sl < current_sl:
                        current_sl = trail_sl

                if use_be and not be_activated and not trail_activated and be_level_sell is not None and l <= be_level_sell:
                    current_sl = entry - 1.0
                    be_activated = True

        last_c = closes[-1]
        pnl = (last_c - entry) if direction == 'BUY' else (entry - last_c)
        return 'EOD', last_c, times[-1], pnl

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

        decisive = wins + losses
        win_rate = (wins / decisive * 100) if decisive > 0 else 0
        total_pnl = df['pnl_pts'].sum()
        total_pnl_dollars = df['pnl_dollars'].sum()
        avg_win = df[df['result'] == 'WIN']['pnl_pts'].mean() if wins > 0 else 0
        avg_loss = df[df['result'] == 'LOSS']['pnl_pts'].mean() if losses > 0 else 0
        gross_loss = df[df['pnl_pts'] < 0]['pnl_pts'].sum()
        profit_factor = abs(df[df['pnl_pts'] > 0]['pnl_pts'].sum() / gross_loss) if gross_loss != 0 else float('inf')

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

        first_date = df['trade_date'].min()
        last_date = df['trade_date'].max()
        if first_date and last_date:
            total_weeks = max(1, (last_date - first_date).days / 7)
            trades_per_week = total / total_weeks
        else:
            trades_per_week = 0

        return {
            'total_trades': total,
            'wins': wins,
            'losses': losses,
            'breakevens': bes,
            'eod_exits': eods,
            'decisive_trades': decisive,
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
            'trades_per_week': round(trades_per_week, 2),
        }

    def _empty_metrics(self):
        keys = [
            'total_trades', 'wins', 'losses', 'breakevens', 'eod_exits',
            'decisive_trades', 'win_rate', 'total_pnl_pts', 'total_pnl_dollars',
            'avg_win_pts', 'avg_loss_pts', 'profit_factor',
            'max_drawdown_pts', 'max_drawdown_dollars', 'avg_rr_on_wins',
            'best_day_pts', 'worst_day_pts', 'avg_daily_pnl',
            'winning_days', 'losing_days', 'total_trading_days',
            'max_consecutive_wins', 'max_consecutive_losses',
            'trades_per_month', 'trades_per_week',
        ]
        return {k: 0 for k in keys}
