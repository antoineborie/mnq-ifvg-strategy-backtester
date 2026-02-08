import pandas as pd
import numpy as np
from datetime import date


US_ECONOMIC_EVENTS = [
    {"date": "2025-01-10", "event": "NFP", "category": "employment", "impact": "high"},
    {"date": "2025-01-14", "event": "PPI", "category": "inflation", "impact": "high"},
    {"date": "2025-01-15", "event": "CPI", "category": "inflation", "impact": "high"},

    {"date": "2025-02-07", "event": "NFP", "category": "employment", "impact": "high"},
    {"date": "2025-02-12", "event": "CPI", "category": "inflation", "impact": "high"},
    {"date": "2025-02-13", "event": "PPI", "category": "inflation", "impact": "high"},

    {"date": "2025-03-07", "event": "NFP", "category": "employment", "impact": "high"},
    {"date": "2025-03-12", "event": "CPI", "category": "inflation", "impact": "high"},
    {"date": "2025-03-13", "event": "PPI", "category": "inflation", "impact": "high"},

    {"date": "2025-04-04", "event": "NFP", "category": "employment", "impact": "high"},
    {"date": "2025-04-10", "event": "CPI", "category": "inflation", "impact": "high"},
    {"date": "2025-04-11", "event": "PPI", "category": "inflation", "impact": "high"},

    {"date": "2025-05-02", "event": "NFP", "category": "employment", "impact": "high"},
    {"date": "2025-05-13", "event": "CPI", "category": "inflation", "impact": "high"},
    {"date": "2025-05-14", "event": "PPI", "category": "inflation", "impact": "high"},
]

US_BANK_HOLIDAYS_2025 = [
    {"date": "2025-01-01", "name": "New Year's Day"},
    {"date": "2025-01-09", "name": "Carter Mourning Day"},
    {"date": "2025-01-20", "name": "MLK Day"},
    {"date": "2025-02-17", "name": "Presidents' Day"},
    {"date": "2025-04-18", "name": "Good Friday"},
    {"date": "2025-05-26", "name": "Memorial Day"},
]


def get_events_df():
    df = pd.DataFrame(US_ECONOMIC_EVENTS)
    df['date'] = pd.to_datetime(df['date']).dt.date
    return df


def get_holidays_df():
    df = pd.DataFrame(US_BANK_HOLIDAYS_2025)
    df['date'] = pd.to_datetime(df['date']).dt.date
    return df


def classify_trading_day(trade_date):
    if isinstance(trade_date, pd.Timestamp):
        trade_date = trade_date.date()

    events_df = get_events_df()
    holidays_df = get_holidays_df()

    day_events = events_df[events_df['date'] == trade_date]
    is_holiday = trade_date in holidays_df['date'].values

    pre_holiday = False
    for _, h in holidays_df.iterrows():
        delta = (h['date'] - trade_date).days
        if 1 <= delta <= 1:
            pre_holiday = True

    post_holiday = False
    for _, h in holidays_df.iterrows():
        delta = (trade_date - h['date']).days
        if 1 <= delta <= 1:
            post_holiday = True

    tags = []
    if is_holiday:
        tags.append("HOLIDAY")
    if not day_events.empty:
        for _, evt in day_events.iterrows():
            tags.append(evt['event'])
    if pre_holiday:
        tags.append("PRE_HOLIDAY")
    if post_holiday:
        tags.append("POST_HOLIDAY")

    if not tags:
        tags.append("NORMAL")

    return tags


def analyze_event_impact(trades_df):
    if trades_df.empty:
        return {}

    trades = trades_df.copy()
    if 'trade_date' not in trades.columns:
        trades['trade_date'] = pd.to_datetime(trades['entry_time']).dt.date

    trades['day_tags'] = trades['trade_date'].apply(classify_trading_day)

    all_categories = set()
    for tags in trades['day_tags']:
        all_categories.update(tags)

    report = {}

    for cat in sorted(all_categories):
        mask = trades['day_tags'].apply(lambda x: cat in x)
        subset = trades[mask]
        if subset.empty:
            continue

        total = len(subset)
        wins = len(subset[subset['result'] == 'WIN'])
        losses = len(subset[subset['result'] == 'LOSS'])
        bes = len(subset[subset['result'] == 'BE'])
        pnl = subset['pnl_pts'].sum()
        avg_pnl = subset['pnl_pts'].mean()
        win_rate = (wins / total * 100) if total > 0 else 0
        gross_profit = subset[subset['pnl_pts'] > 0]['pnl_pts'].sum()
        gross_loss = abs(subset[subset['pnl_pts'] < 0]['pnl_pts'].sum())
        pf = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')

        unique_days = subset['trade_date'].nunique()

        report[cat] = {
            'trades': total,
            'trading_days': unique_days,
            'wins': wins,
            'losses': losses,
            'breakevens': bes,
            'win_rate': round(win_rate, 1),
            'total_pnl': round(pnl, 2),
            'avg_pnl': round(avg_pnl, 2),
            'profit_factor': round(pf, 2),
            'avg_trades_per_day': round(total / unique_days, 1) if unique_days > 0 else 0,
        }

    return report


def get_event_recommendations(report):
    recommendations = []

    for cat, stats in report.items():
        if cat == 'NORMAL':
            continue

        if stats['trades'] < 3:
            recommendations.append({
                'event': cat,
                'verdict': 'INSUFFICIENT DATA',
                'detail': f"Only {stats['trades']} trades - need more data",
                'color': 'gray',
            })
            continue

        normal = report.get('NORMAL', {})
        normal_pf = normal.get('profit_factor', 1.0)
        normal_wr = normal.get('win_rate', 50.0)
        normal_avg = normal.get('avg_pnl', 0)

        pf_diff = stats['profit_factor'] - normal_pf
        wr_diff = stats['win_rate'] - normal_wr
        avg_diff = stats['avg_pnl'] - normal_avg

        score = 0
        if stats['profit_factor'] > 1.0:
            score += 2
        if stats['win_rate'] > normal_wr:
            score += 1
        if stats['avg_pnl'] > 0:
            score += 2
        if pf_diff > 0.2:
            score += 1
        if stats['profit_factor'] < 0.7:
            score -= 3
        if stats['avg_pnl'] < -10:
            score -= 2

        if score >= 3:
            verdict = "TRADE"
            color = "green"
            detail = f"PF {stats['profit_factor']:.2f} | WR {stats['win_rate']}% | Avg {stats['avg_pnl']:+.1f} pts"
        elif score >= 1:
            verdict = "NEUTRAL"
            color = "orange"
            detail = f"PF {stats['profit_factor']:.2f} | WR {stats['win_rate']}% | Mixed signals"
        else:
            verdict = "AVOID"
            color = "red"
            detail = f"PF {stats['profit_factor']:.2f} | WR {stats['win_rate']}% | Avg {stats['avg_pnl']:+.1f} pts"

        recommendations.append({
            'event': cat,
            'verdict': verdict,
            'detail': detail,
            'color': color,
            'stats': stats,
        })

    return recommendations
