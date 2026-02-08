import pandas as pd
import numpy as np
from scipy import stats as scipy_stats


def perform_full_stat_analysis(df):
    if df is None or len(df) == 0:
        return _empty_results()

    df = df.copy()

    if 'entry_time' in df.columns:
        df['entry_time'] = pd.to_datetime(df['entry_time'])
        if hasattr(df['entry_time'].dt, 'tz') and df['entry_time'].dt.tz is not None:
            entry_naive = df['entry_time'].dt.tz_localize(None)
        else:
            entry_naive = df['entry_time']
        df['trade_date'] = entry_naive.dt.date
        df['year'] = entry_naive.dt.year
        df['month'] = entry_naive.dt.month
        df['year_month'] = entry_naive.dt.to_period('M').astype(str)
        df['day_of_week'] = entry_naive.dt.day_name()
    elif 'trade_date' in df.columns:
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df['year'] = df['trade_date'].dt.year
        df['month'] = df['trade_date'].dt.month
        df['year_month'] = df['trade_date'].dt.to_period('M').astype(str)
        df['day_of_week'] = df['trade_date'].dt.day_name()

    is_win = df['result'] == 'WIN'

    results = {}
    results['cohort'] = _cohort_analysis(df, is_win)
    results['robustness'] = _robustness_analysis(df, is_win)
    results['streaks'] = _streak_analysis(df)
    results['risk_reward'] = _risk_reward_analysis(df, is_win)
    results['volatility'] = _volatility_analysis(df)

    return results


def _empty_results():
    return {
        'cohort': {},
        'robustness': {},
        'streaks': {},
        'risk_reward': {},
        'volatility': {},
    }


def _cohort_analysis(df, is_win):
    total = len(df)
    wins = is_win.sum()
    global_wr = round(wins / total * 100, 2) if total > 0 else 0

    yearly_groups = df.groupby('year')
    yearly_data = []
    for year, g in yearly_groups:
        n = len(g)
        w = (g['result'] == 'WIN').sum()
        yearly_data.append({
            'year': year,
            'trades': n,
            'wins': int(w),
            'win_rate': round(w / n * 100, 2) if n > 0 else 0,
            'pnl': round(float(g['pnl_pts'].sum()), 2),
            'avg_pnl': round(float(g['pnl_pts'].mean()), 2),
        })
    yearly = pd.DataFrame(yearly_data)

    monthly_groups = df.groupby('year_month')
    monthly_data = []
    for ym, g in monthly_groups:
        n = len(g)
        w = (g['result'] == 'WIN').sum()
        monthly_data.append({
            'year_month': ym,
            'trades': n,
            'wins': int(w),
            'win_rate': round(w / n * 100, 2) if n > 0 else 0,
            'pnl': round(float(g['pnl_pts'].sum()), 2),
        })
    monthly = pd.DataFrame(monthly_data)

    dow_groups = df.groupby('day_of_week')
    dow_data = []
    for dow, g in dow_groups:
        n = len(g)
        w = (g['result'] == 'WIN').sum()
        dow_data.append({
            'day_of_week': dow,
            'trades': n,
            'wins': int(w),
            'win_rate': round(w / n * 100, 2) if n > 0 else 0,
            'pnl': round(float(g['pnl_pts'].sum()), 2),
        })
    by_dow = pd.DataFrame(dow_data)
    dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    by_dow['day_of_week'] = pd.Categorical(by_dow['day_of_week'], categories=dow_order, ordered=True)
    by_dow = by_dow.sort_values('day_of_week').reset_index(drop=True)

    yearly_wrs = yearly['win_rate'].values
    equity_drift = 0
    if len(yearly_wrs) >= 2:
        equity_drift = round(float(yearly_wrs[-1] - yearly_wrs[0]), 2)

    monthly_wrs = monthly['win_rate'].values
    wr_stability = round(float(np.std(monthly_wrs)), 2) if len(monthly_wrs) > 1 else 0

    qualified_monthly = monthly[monthly['trades'] >= 3]
    q_wrs = qualified_monthly['win_rate'].values if len(qualified_monthly) > 0 else np.array([])

    months_below_60 = int((q_wrs < 60).sum()) if len(q_wrs) > 0 else 0
    months_below_55 = int((q_wrs < 55).sum()) if len(q_wrs) > 0 else 0
    months_below_50 = int((q_wrs < 50).sum()) if len(q_wrs) > 0 else 0
    monthly_wr_floor = round(float(q_wrs.min()), 1) if len(q_wrs) > 0 else 0
    monthly_wr_ceiling = round(float(q_wrs.max()), 1) if len(q_wrs) > 0 else 0
    months_at_target = int((q_wrs >= 60).sum()) if len(q_wrs) > 0 else 0
    total_qualified = len(q_wrs)
    consistency_pct = round(months_at_target / total_qualified * 100, 1) if total_qualified > 0 else 0

    neg_pnl_months = 0
    if len(qualified_monthly) > 0:
        neg_pnl_months = int((qualified_monthly['pnl'].values < 0).sum())

    return {
        'global_win_rate': global_wr,
        'total_trades': total,
        'total_wins': int(wins),
        'yearly': yearly.to_dict('records'),
        'monthly': monthly.to_dict('records'),
        'by_day_of_week': by_dow.to_dict('records'),
        'equity_drift': equity_drift,
        'wr_stability_std': wr_stability,
        'months_below_60': months_below_60,
        'months_below_55': months_below_55,
        'months_below_50': months_below_50,
        'monthly_wr_floor': monthly_wr_floor,
        'monthly_wr_ceiling': monthly_wr_ceiling,
        'months_at_target': months_at_target,
        'total_qualified_months': total_qualified,
        'consistency_pct': consistency_pct,
        'negative_pnl_months': neg_pnl_months,
    }


def _robustness_analysis(df, is_win):
    total = len(df)
    pnl = df['pnl_pts'].values

    gross_profit = float(pnl[pnl > 0].sum()) if (pnl > 0).any() else 0
    gross_loss = float(abs(pnl[pnl < 0].sum())) if (pnl < 0).any() else 0
    profit_factor = round(gross_profit / gross_loss, 3) if gross_loss > 0 else float('inf')

    expectancy = round(float(pnl.mean()), 3) if total > 0 else 0

    wins_arr = is_win.astype(int).values
    n = len(wins_arr)
    n_wins = int(wins_arr.sum())
    n_losses = n - n_wins

    runs = 1
    for i in range(1, n):
        if wins_arr[i] != wins_arr[i - 1]:
            runs += 1

    expected_runs = 0.0
    if n_wins == 0 or n_losses == 0 or n < 3:
        z_score = 0.0
        z_p_value = 1.0
        z_interpretation = "Insufficient data"
    else:
        expected_runs = 1 + (2 * n_wins * n_losses) / n
        variance = (2 * n_wins * n_losses * (2 * n_wins * n_losses - n)) / (n * n * (n - 1))
        if variance > 0:
            std_runs = np.sqrt(variance)
            z_score = round((runs - expected_runs) / std_runs, 3)
            z_p_value = round(2 * (1 - scipy_stats.norm.cdf(abs(z_score))), 4)
        else:
            z_score = 0.0
            z_p_value = 1.0

        if abs(z_score) < 1.96:
            z_interpretation = "Random (no significant pattern detected)"
        elif z_score > 1.96:
            z_interpretation = "Alternating pattern detected (wins tend to follow losses)"
        else:
            z_interpretation = "Clustering detected (wins/losses tend to streak)"

    kelly_fraction = 0
    avg_win = float(pnl[pnl > 0].mean()) if (pnl > 0).any() else 0
    avg_loss_abs = float(abs(pnl[pnl < 0].mean())) if (pnl < 0).any() else 1
    win_rate_dec = n_wins / n if n > 0 else 0
    if avg_loss_abs > 0:
        b = avg_win / avg_loss_abs
        kelly_fraction = round((win_rate_dec * b - (1 - win_rate_dec)) / b, 4) if b > 0 else 0

    return {
        'profit_factor': profit_factor,
        'expectancy_per_trade': expectancy,
        'gross_profit': round(gross_profit, 2),
        'gross_loss': round(gross_loss, 2),
        'z_score': z_score,
        'z_p_value': z_p_value,
        'z_interpretation': z_interpretation,
        'runs_count': runs,
        'expected_runs': round(expected_runs, 1) if n_wins > 0 and n_losses > 0 else 0,
        'kelly_fraction': kelly_fraction,
        'avg_win': round(avg_win, 2),
        'avg_loss': round(avg_loss_abs, 2),
    }


def _streak_analysis(df):
    results = df['result'].values
    pnl = df['pnl_pts'].values

    max_win_streak = 0
    max_loss_streak = 0
    current_win = 0
    current_loss = 0

    win_streaks = []
    loss_streaks = []

    for r in results:
        if r == 'WIN':
            current_win += 1
            if current_loss > 0:
                loss_streaks.append(current_loss)
            current_loss = 0
        elif r == 'LOSS':
            current_loss += 1
            if current_win > 0:
                win_streaks.append(current_win)
            current_win = 0
        else:
            if current_win > 0:
                win_streaks.append(current_win)
            if current_loss > 0:
                loss_streaks.append(current_loss)
            current_win = 0
            current_loss = 0

    if current_win > 0:
        win_streaks.append(current_win)
    if current_loss > 0:
        loss_streaks.append(current_loss)

    max_win_streak = max(win_streaks) if win_streaks else 0
    max_loss_streak = max(loss_streaks) if loss_streaks else 0
    avg_win_streak = round(np.mean(win_streaks), 2) if win_streaks else 0
    avg_loss_streak = round(np.mean(loss_streaks), 2) if loss_streaks else 0

    cum_pnl = np.cumsum(pnl)
    peak = np.maximum.accumulate(cum_pnl)
    drawdown = cum_pnl - peak
    max_dd = float(np.min(drawdown)) if len(drawdown) > 0 else 0

    max_dd_idx = int(np.argmin(drawdown)) if len(drawdown) > 0 else 0
    peak_idx = int(np.argmax(cum_pnl[:max_dd_idx + 1])) if max_dd_idx > 0 else 0
    dd_duration = max_dd_idx - peak_idx

    recovery_idx = None
    peak_val = cum_pnl[peak_idx] if peak_idx < len(cum_pnl) else 0
    for j in range(max_dd_idx, len(cum_pnl)):
        if cum_pnl[j] >= peak_val:
            recovery_idx = j
            break
    recovery_trades = (recovery_idx - max_dd_idx) if recovery_idx is not None else None

    worst_streak_pnl = 0
    if loss_streaks:
        idx = 0
        for streak_len in loss_streaks:
            while idx < len(results) and results[idx] != 'LOSS':
                idx += 1
            streak_pnl = float(pnl[idx:idx + streak_len].sum())
            if streak_pnl < worst_streak_pnl:
                worst_streak_pnl = streak_pnl
            idx += streak_len

    return {
        'max_win_streak': max_win_streak,
        'max_loss_streak': max_loss_streak,
        'avg_win_streak': avg_win_streak,
        'avg_loss_streak': avg_loss_streak,
        'win_streak_distribution': _count_distribution(win_streaks),
        'loss_streak_distribution': _count_distribution(loss_streaks),
        'max_drawdown_pts': round(max_dd, 2),
        'drawdown_duration_trades': dd_duration,
        'recovery_trades': recovery_trades,
        'worst_loss_streak_pnl': round(worst_streak_pnl, 2),
    }


def _count_distribution(streaks):
    if not streaks:
        return {}
    dist = {}
    for s in streaks:
        dist[s] = dist.get(s, 0) + 1
    return dict(sorted(dist.items()))


def _risk_reward_analysis(df, is_win):
    total = len(df)
    if total == 0:
        return {}

    wins_df = df[df['result'] == 'WIN']
    losses_df = df[df['result'] == 'LOSS']

    if 'rr_achieved' in df.columns:
        avg_rr_wins = round(float(wins_df['rr_achieved'].mean()), 3) if len(wins_df) > 0 else 0
        avg_rr_losses = round(float(losses_df['rr_achieved'].mean()), 3) if len(losses_df) > 0 else 0
    else:
        avg_rr_wins = 0
        avg_rr_losses = 0

    avg_win_pts = float(wins_df['pnl_pts'].mean()) if len(wins_df) > 0 else 0
    avg_loss_pts = float(abs(losses_df['pnl_pts'].mean())) if len(losses_df) > 0 else 1

    real_rr_ratio = round(avg_win_pts / avg_loss_pts, 3) if avg_loss_pts > 0 else float('inf')

    breakeven_wr = round(1 / (1 + real_rr_ratio) * 100, 2) if real_rr_ratio > 0 else 50.0

    observed_wr = round(is_win.sum() / total * 100, 2)

    edge = round(observed_wr - breakeven_wr, 2)

    if 'risk_pts' in df.columns:
        avg_risk = round(float(df['risk_pts'].mean()), 2)
        median_risk = round(float(df['risk_pts'].median()), 2)
        risk_std = round(float(df['risk_pts'].std()), 2) if len(df) > 1 else 0
    else:
        avg_risk = 0
        median_risk = 0
        risk_std = 0

    expectancy_r = 0
    if total > 0 and avg_loss_pts > 0:
        win_pct = is_win.sum() / total
        loss_pct = 1 - win_pct
        expectancy_r = round(win_pct * real_rr_ratio - loss_pct, 3)

    return {
        'real_rr_ratio': real_rr_ratio,
        'avg_rr_wins': avg_rr_wins,
        'avg_rr_losses': avg_rr_losses,
        'avg_win_pts': round(avg_win_pts, 2),
        'avg_loss_pts': round(avg_loss_pts, 2),
        'breakeven_win_rate': breakeven_wr,
        'observed_win_rate': observed_wr,
        'edge_over_breakeven': edge,
        'avg_risk_pts': avg_risk,
        'median_risk_pts': median_risk,
        'risk_consistency_std': risk_std,
        'expectancy_r': expectancy_r,
    }


def _volatility_analysis(df):
    if len(df) < 5:
        return {
            'sharpe_ratio': 0, 'sortino_ratio': 0, 'calmar_ratio': 0,
            'pnl_std': 0, 'pnl_skew': 0, 'pnl_kurtosis': 0,
            'monthly_returns': [],
        }

    pnl = df['pnl_pts'].values

    mean_pnl = float(np.mean(pnl))
    std_pnl = float(np.std(pnl, ddof=1)) if len(pnl) > 1 else 1

    sharpe = round(mean_pnl / std_pnl, 3) if std_pnl > 0 else 0

    downside = pnl[pnl < 0]
    downside_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else float(np.std(pnl, ddof=1)) if len(pnl) > 1 else 1
    sortino = round(mean_pnl / downside_std, 3) if downside_std > 0 else 0

    cum_pnl = np.cumsum(pnl)
    peak = np.maximum.accumulate(cum_pnl)
    max_dd = float(np.min(cum_pnl - peak))
    total_return = float(cum_pnl[-1])

    if 'entry_time' in df.columns:
        first = pd.to_datetime(df['entry_time'].iloc[0])
        last = pd.to_datetime(df['entry_time'].iloc[-1])
        years = max((last - first).days / 365.25, 0.1)
    else:
        years = max(len(df) / 252, 0.1)

    annualized_return = total_return / years
    calmar = round(annualized_return / abs(max_dd), 3) if max_dd != 0 else 0

    skew = round(float(scipy_stats.skew(pnl)), 3) if len(pnl) > 2 else 0
    kurt = round(float(scipy_stats.kurtosis(pnl)), 3) if len(pnl) > 3 else 0

    monthly_returns = []
    if 'year_month' in df.columns:
        monthly_pnl = df.groupby('year_month')['pnl_pts'].sum()
        monthly_returns = [
            {'month': str(m), 'pnl': round(float(v), 2)}
            for m, v in monthly_pnl.items()
        ]

    daily_sharpe = 0
    if 'trade_date' in df.columns:
        daily_pnl = df.groupby('trade_date')['pnl_pts'].sum()
        if len(daily_pnl) > 5:
            d_mean = float(daily_pnl.mean())
            d_std = float(daily_pnl.std())
            if d_std > 0:
                daily_sharpe = round(d_mean / d_std * np.sqrt(252), 3)

    return {
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'calmar_ratio': calmar,
        'daily_sharpe_annualized': daily_sharpe,
        'pnl_std': round(std_pnl, 3),
        'pnl_skew': skew,
        'pnl_kurtosis': kurt,
        'mean_pnl_per_trade': round(mean_pnl, 3),
        'total_return': round(total_return, 2),
        'annualized_return': round(annualized_return, 2),
        'max_drawdown': round(max_dd, 2),
        'monthly_returns': monthly_returns,
    }
