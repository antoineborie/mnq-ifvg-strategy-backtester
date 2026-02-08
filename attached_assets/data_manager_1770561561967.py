import pandas as pd
import numpy as np

class DataMaster:
    def __init__(self, raw_df):
        self.df = raw_df.sort_index()
        self.cumulative_pnl = 0
        self.trade_log = []
        self.vault = []
        self.MAX_RANGE_RATIO = 2.2  # Filtre 1 : Évite l'épuisement
        self.BE_RR_TRIGGER = 1.1    # Filtre 4 : Se met au BE à 1.1 de RR

        if self.df.index.tz is None:
            self.df = self.df.tz_localize('UTC').tz_convert('America/New_York')
        else: 
            self.df = self.df.tz_convert('America/New_York')

    def get_market_context(self, target_date):
        target_date_obj = pd.Timestamp(target_date).date()
        day_data = self.df[self.df.index.date == target_date_obj]
        if day_data.empty: return None, None, None
        
        main_symbol = day_data.groupby('symbol')['volume'].sum().idxmax()
        clean_df = self.df[self.df['symbol'] == main_symbol]
        
        history = clean_df[clean_df.index.date < target_date_obj]
        session = clean_df[clean_df.index.date == target_date_obj]
        return history, session, main_symbol

    def calculate_ict_levels(self, history, session):
        levels = {'mid': None, 'pdh': None, 'pdl': None, 'major_highs': [], 'major_lows': []}
        if session.empty: 
            return levels

        levels['mid'] = float(session.iloc[0]['open'])
        
        if not history.empty:
            last_day_date = history.index.normalize().unique()[-1]
            last_day_data = history[history.index.normalize() == last_day_date]
            levels['pdh'] = float(last_day_data['high'].max())
            levels['pdl'] = float(last_day_data['low'].min())

        am_data = session.between_time('02:00', '09:20')
        am_highs, am_lows = [], []
        
        if not am_data.empty:
            am_highs.append(float(am_data['high'].max()))
            am_lows.append(float(am_data['low'].min()))
            
            am_m15 = am_data.resample('15min').agg({'high': 'max', 'low': 'min'})
            for i in range(1, len(am_m15)-1):
                if am_m15['high'].iloc[i] > am_m15['high'].iloc[i-1] and am_m15['high'].iloc[i] > am_m15['high'].iloc[i+1]:
                    am_highs.append(float(am_m15['high'].iloc[i]))
                if am_m15['low'].iloc[i] < am_m15['low'].iloc[i-1] and am_m15['low'].iloc[i] < am_m15['low'].iloc[i+1]:
                    am_lows.append(float(am_m15['low'].iloc[i]))

        recent_h1 = history.tail(48).resample('1h').agg({'high': 'max', 'low': 'min'}).dropna()
        h1_highs, h1_lows = [], []
        if len(recent_h1) > 5:
            for i in range(2, len(recent_h1)-2):
                if recent_h1['high'].iloc[i] == recent_h1['high'].iloc[i-2:i+3].max():
                    h1_highs.append(float(recent_h1['high'].iloc[i]))
                if recent_h1['low'].iloc[i] == recent_h1['low'].iloc[i-2:i+3].min():
                    h1_lows.append(float(recent_h1['low'].iloc[i]))

        def finalize(prices, reverse=True):
            if not prices: return []
            prices = sorted(list(set(prices)), reverse=reverse)
            final = []
            for p in prices:
                if not any(abs(p - f) < 8.0 for f in final):
                    final.append(p)
            return final[:8]

        levels['major_highs'] = finalize(am_highs + h1_highs + ([levels['pdh']] if levels['pdh'] else []), True)
        levels['major_lows'] = finalize(am_lows + h1_lows + ([levels['pdl']] if levels['pdl'] else []), False)
        
        levels['old_highs'], levels['old_lows'] = levels['major_highs'], levels['major_lows']
        
        return levels

    def detect_signals(self, session, levels, tf='1min'):
        signals = []
        if session.empty: return signals
        
        # --- 1. ANALYSE DU FLUX & CONTEXTE ---
        h1_data = session.resample('h').agg({'open':'first', 'close':'last'}).dropna()
        if len(h1_data) < 2: return signals
        last_h1 = h1_data.iloc[-1]
        h1_bias = "BULL" if last_h1['close'] > last_h1['open'] else "BEAR"

        po3 = self.analyze_po3(None, session)
        # On autorise le trading même en forte expansion (jusqu'à 4.0) pour ne pas rater les rallyes
        if po3['range_ratio'] > 4.0: return signals
        
        # L'expansion est confirmée si ratio > 1.5
        is_expansion = po3['range_ratio'] > 1.5

        # --- PARAMÈTRES OPTIMISÉS V3 ---
        MIN_FVG = 3.5            # Plus sensible pour le Nasdaq (réduit de 4.0)
        MAX_RISK_PTS = 55        # Protection large pour volatilité (augmenté de 50)
        COOLDOWN = 15            # Agression maximale (réduit de 20)
        
        midnight_open = session.iloc[0]['open']
        trades_count = 0
        last_trade_end_time = None 
        
        session_tf = session.resample(tf).agg({
            'open':'first','high':'max','low':'min','close':'last', 'volume':'sum'
        }).dropna()

        opening_range = session_tf.between_time("09:30", "09:35")
        or_high = opening_range['high'].max() if not opening_range.empty else None
        or_low = opening_range['low'].min() if not opening_range.empty else None

        for i in range(5, len(session_tf) - 2):
            timestamp = session_tf.index[i]
            
            if not ("09:30" <= timestamp.strftime('%H:%M') <= "10:45"): continue
            if trades_count >= 2: break
            
            if last_trade_end_time is not None:
                if timestamp <= (last_trade_end_time + pd.Timedelta(minutes=COOLDOWN)):
                    continue
            
            curr, prev, old = session_tf.iloc[i], session_tf.iloc[i-1], session_tf.iloc[i-2]
            found_trade = False

            # --- 2. LOGIQUE DE FLUX UNIQUEMENT (ADIEU REVERSAL) ---
            
            # Cas SELL : Biais H1 Baissier + Prix sous Midnight Open (Confirmation de distribution)
            if h1_bias == "BEAR" and curr['close'] < midnight_open:
                sell_levels = levels['major_highs'] + ([or_high] if or_high else [])
                
                # Condition 1: Sweep classique OU Condition 2: Cassure en Expansion (Continuation)
                swept = any((session_tf.iloc[i-3:i]['high'] > lvl).any() for lvl in sell_levels)                
                displacement = curr['close'] < prev['low']
                fvg = (old['low'] - curr['high']) >= MIN_FVG
                
                # En expansion, on accepte le trade sans sweep si le displacement est fort
                if (swept or is_expansion) and fvg and displacement:
                    risk = abs(max(prev['high'], old['high']) - curr['close']) + 2.0
                    if risk <= MAX_RISK_PTS:
                        entry = round(curr['close'], 2)
                        sl = round(entry + risk, 2)
                        lows = [l for l in levels['major_lows'] if l < entry]
                        tp = max(lows) if lows else round(entry - (risk * 2.5), 2)
                        
                        if (entry - tp) / risk >= 1.5:
                            res, pnl, exit_t = self._evaluate_trade(session_tf.iloc[i+1:], entry, tp, sl, "SELL")
                            s_data = {'time': timestamp, 'type': "🔴 SELL [FLUX/CONT]", 
                                    'entry': entry, 'tp': tp, 'sl': sl, 'res': res, 'pnl': pnl}
                            signals.append(s_data); self.trade_log.append(s_data); self.cumulative_pnl += pnl
                            last_trade_end_time = exit_t; trades_count += 1; found_trade = True

            if found_trade: continue

            # Cas BUY : Biais H1 Haussier + Prix au-dessus Midnight Open (Confirmation d'accumulation)
            if h1_bias == "BULL" and curr['close'] > midnight_open:
                buy_levels = levels['major_lows'] + ([or_low] if or_low else [])
                
                swept = any((session_tf.iloc[i-3:i]['low'] < lvl).any() for lvl in buy_levels)                
                displacement = curr['close'] > prev['high']
                fvg = (curr['low'] - old['high']) >= MIN_FVG
                
                if (swept or is_expansion) and fvg and displacement:
                    risk = abs(curr['close'] - min(prev['low'], old['low'])) + 2.0
                    if risk <= MAX_RISK_PTS:
                        entry = round(curr['close'], 2)
                        sl = round(entry - risk, 2)
                        highs = [l for l in levels['major_highs'] if l > entry]
                        tp = min(highs) if highs else round(entry + (risk * 2.5), 2)
                        
                        if (tp - entry) / risk >= 1.5:
                            res, pnl, exit_t = self._evaluate_trade(session_tf.iloc[i+1:], entry, tp, sl, "BUY")
                            s_data = {'time': timestamp, 'type': "🟢 BUY [FLUX/CONT]", 
                                    'entry': entry, 'tp': tp, 'sl': sl, 'res': res, 'pnl': pnl}
                            signals.append(s_data); self.trade_log.append(s_data); self.cumulative_pnl += pnl
                            last_trade_end_time = exit_t; trades_count += 1

        return signals

    def _evaluate_trade(self, future_data, entry, tp, sl, direction):
        if future_data.empty: 
            return "🕒 EOD", 0, None
        
        current_sl = sl
        
        is_be_active = False 

        for timestamp, row in future_data.iterrows():
            high, low, close = row['high'], row['low'], row['close']

            if direction == "BUY":
                if high >= tp:
                    return "💰 💰 WIN", round(tp - entry, 2), timestamp
                
                if low <= current_sl:
                    return "❌ LOSS", round(sl - entry, 2), timestamp
            
            else:
                if low <= tp:
                    return "💰 💰 WIN", round(entry - tp, 2), timestamp
                
                if high >= current_sl:
                    return "❌ LOSS", round(entry - sl, 2), timestamp

        final_pnl = round(close - entry if direction == "BUY" else entry - close, 2)
        res_eod = "🕒 EOD (PROFIT)" if final_pnl > 0 else "🕒 EOD (LOSS)"
        return res_eod, final_pnl, future_data.index[-1]

    def generate_session_report(self):
        if not self.trade_log: 
            return "Aucun trade exécuté aujourd'hui."
            
        df = pd.DataFrame(self.trade_log)
        
        wins = len(df[(df['res'].str.contains('WIN', na=False)) | (df['pnl'] > 0)])
        
        total_trades = len(df)
        winrate = round((wins / total_trades) * 100) if total_trades > 0 else 0
        
        session_pnl = round(df['pnl'].sum(), 2)
        
        rep = f"\n💰 P&L Session : {session_pnl} pts | 🏆 Winrate : {winrate}%\n"
        
        self.trade_log = [] 
        return rep

    def analyze_po3(self, h, s):
        pm = s.between_time('00:00', '09:30')
        if pm.empty: return {"is_po3": False, "status": "No Data", "range_ratio": 0, "vol_ratio": 0}
        curr_r = pm['high'].max() - pm['low'].min()
        return {"is_po3": curr_r < 80, "status": "PO3 VALIDE" if curr_r < 80 else "Expansion", "range_ratio": round(curr_r/100, 2), "vol_ratio": 1.0}

    def collect_daily_results(self, daily_signals):
        """
        À appeler dans la boucle journalière pour accumuler les trades.
        """
        if daily_signals:
            for s in daily_signals:
                self.vault.append(s)
        return len(self.vault)

    def get_full_risk_analysis(self):
        if not hasattr(self, 'vault') or not self.vault: 
            return print("❌ Aucun trade dans le coffre-fort (vault) pour l'analyse.")
            
        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy as np

        df = pd.DataFrame(self.vault)
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values('time') 
        df['cum_pnl'] = df['pnl'].cumsum()
        
        df['peak'] = df['cum_pnl'].cummax()
        df['dd'] = df['cum_pnl'] - df['peak']
        max_dd = df['dd'].min()
        
        print(f"\n☢️  === RAPPORT DE SURVIE & PERFORMANCE ===")
        print(f"├─ TOTAL TRADES ANALYSÉS : {len(df)}")
        print(f"├─ PROFIT TOTAL          : {df['cum_pnl'].iloc[-1]:.2f} pts")
        print(f"├─ DRAWDOWN MAX          : {max_dd:.2f} pts")
        print(f"└─ WINRATE GLOBAL        : {(df['pnl'] > 0).mean()*100:.2f}%")

        plt.style.use('dark_background') # Optionnel pour un look "TradingView"
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, 
                                       gridspec_kw={'height_ratios': [3, 1]})

        ax1.plot(df['time'], df['cum_pnl'], color='#00ff88', lw=2, label='PnL Cumulé')
        ax1.fill_between(df['time'], df['cum_pnl'], 0, 
                         where=(df['cum_pnl'] >= 0), color='#00ff88', alpha=0.2)
        ax1.fill_between(df['time'], df['cum_pnl'], 0, 
                         where=(df['cum_pnl'] < 0), color='#ff4d4d', alpha=0.2)
        
        ax1.set_title("COURBE DE PERFORMANCE GLOBALE", fontsize=14, fontweight='bold')
        ax1.set_ylabel("Points")
        ax1.grid(True, alpha=0.2)
        ax1.legend(loc='upper left')

        ax2.fill_between(df['time'], df['dd'], 0, color='#ff4d4d', alpha=0.6, label='Drawdown (Points)')
        ax2.set_title("INTENSITÉ DU DRAWDOWN (RISQUE)", fontsize=12)
        ax2.set_ylabel("Baisse")
        ax2.grid(True, alpha=0.2)
        ax2.legend(loc='lower left')

        plt.tight_layout()
        plt.show()