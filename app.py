import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

from data_loader import list_data_files, load_data, get_active_contract
from ifvg_strategy import IFVGStrategy
from econ_calendar import analyze_event_impact, get_event_recommendations, get_events_df, get_holidays_df
from optimizer import run_optimization, PARAM_GRID, FIXED_PARAMS, get_param_grid_size
from stat_analysis import perform_full_stat_analysis
import json

st.set_page_config(
    page_title="MNQ IFVG Backtester",
    page_icon="📊",
    layout="wide",
)

st.markdown("""
<style>
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #2d3561;
        border-radius: 10px;
        padding: 15px;
    }
    .verdict-trade { color: #00d4aa; font-weight: bold; font-size: 18px; }
    .verdict-avoid { color: #ff4757; font-weight: bold; font-size: 18px; }
    .verdict-neutral { color: #ffa502; font-weight: bold; font-size: 18px; }
    .verdict-nodata { color: #888; font-size: 16px; }
</style>
""", unsafe_allow_html=True)

st.title("MNQ Futures - IFVG Strategy Backtester")
st.caption("Multi-Timeframe Inversed Fair Value Gap | H1 Bias → M15 IFVG → M1 Retracement Entry")

data_files = list_data_files()
file_labels = [os.path.basename(f).replace('.pkl', '').replace('MNQ_', '') for f in data_files]

with st.sidebar:
    st.header("Configuration")

    st.subheader("Data Selection")
    if not data_files:
        st.error("No data files found in data/ directory")
        st.stop()

    selected_files = st.multiselect(
        "Select months to backtest",
        options=data_files,
        default=data_files,
        format_func=lambda x: os.path.basename(x).replace('.pkl', '').replace('MNQ_', ''),
    )

    if not selected_files:
        st.warning("Select at least one month")
        st.stop()

    st.subheader("Strategy Parameters")

    min_fvg = st.slider("Min FVG Size (pts)", 1.0, 15.0, 3.0, 0.5,
                         help="Minimum gap size on M15 to qualify as FVG")
    max_fvg_age = st.slider("Max FVG Age (M15 bars)", 2, 24, 15, 1,
                             help="Maximum age of M15 FVG before expiry")
    rr_target = st.slider("Risk:Reward Target", 0.5, 5.0, 1.2, 0.1,
                           help="Target R:R ratio for take profit")
    max_risk = st.slider("Max Risk (pts)", 5.0, 60.0, 25.0, 1.0)
    min_risk = st.slider("Min Risk (pts)", 1.0, 15.0, 5.0, 1.0)
    max_trades = st.slider("Max Trades / Day", 1, 4, 2, 1)
    retracement_pct = st.slider("Retracement % into IFVG zone", 20, 80, 60, 5,
                                 help="How deep price must retrace into the inverted FVG zone on M1")
    cooldown = st.slider("Cooldown (minutes)", 0, 30, 10, 1)
    entry_start_time = st.selectbox("Entry Start Time (ET)", ['09:30', '09:35', '09:40', '09:45', '09:50', '09:55', '10:00'],
                                     index=3, help="Only look for entries after this time (later = higher win rate)")
    structure_lookback = st.slider("Structure Lookback (days)", 5, 60, 20, 5,
                                    help="Days of history for daily/weekly structure levels")

    st.subheader("Killzone (ET)")
    kz_start = st.text_input("Start", "09:30")
    kz_end = st.text_input("End", "11:00")

    st.subheader("Target Mode")
    target_mode = st.radio("Take Profit Method", ["fixed_rr", "ssl"],
                            format_func=lambda x: "Fixed R:R" if x == "fixed_rr" else "SSL/BSL (Liquidity Levels)",
                            help="Fixed R:R uses your R:R target. SSL targets nearest liquidity level.")

    st.subheader("Risk Management")
    use_be = st.checkbox("Breakeven Protection", value=True)
    be_trigger = st.slider("BE Trigger (xR)", 0.3, 2.0, 0.5, 0.1,
                            help="Move stop to breakeven after price moves this many R in your favor") if use_be else 0.5

    st.subheader("Trailing Stop")
    use_trail = st.checkbox("Trailing Stop", value=True)
    if use_trail:
        trail_trigger = st.slider("Trail Trigger (xR)", 0.3, 2.0, 0.5, 0.1,
                                   help="Start trailing after this many R in profit")
        trail_offset = st.slider("Trail Offset (%)", 10, 70, 30, 5,
                                  help="Trail distance as % of risk")
    else:
        trail_trigger = 0.5
        trail_offset = 30

    st.subheader("Quality Filters")
    use_displacement = st.checkbox("Displacement Filter", value=True,
                                    help="Require strong displacement candle behind FVG")
    if use_displacement:
        disp_body_pct = st.slider("Min Displacement Body %", 30, 80, 55, 5)
        disp_size = st.slider("Min Displacement Size (pts)", 1.0, 10.0, 3.5, 0.5)
    else:
        disp_body_pct = 55
        disp_size = 3.5

    use_confirmation = st.checkbox("M1 Confirmation Candle", value=True,
                                    help="Require a rejection/momentum candle on M1 before entry")

    with st.expander("Advanced Filters"):
        use_session_momentum = st.checkbox("Session Momentum Filter", value=False,
                                            help="Require pre-killzone momentum to align with bias")
        momentum_threshold = st.slider("Momentum Threshold", 0.1, 0.8, 0.4, 0.1) if use_session_momentum else 0.4

        use_range_filter = st.checkbox("Previous Day Range Filter", value=False,
                                        help="Filter days by previous day's range")
        min_range = st.slider("Min Prev Day Range", 20.0, 150.0, 60.0, 10.0) if use_range_filter else 60.0
        max_range = st.slider("Max Prev Day Range", 150.0, 800.0, 400.0, 50.0) if use_range_filter else 400.0

        use_confluence = st.checkbox("Structure Confluence", value=False,
                                      help="FVG zone must be near a key structure level")
        confluence_dist = st.slider("Confluence Distance (pts)", 10.0, 80.0, 50.0, 5.0) if use_confluence else 50.0

        use_trend = st.checkbox("Multi-Day Trend Filter", value=False,
                                 help="Require multi-day trend to align with H1 bias")
        trend_days = st.slider("Trend Lookback Days", 2, 10, 3) if use_trend else 3

    contract_value = st.number_input("Point Value ($)", value=2.0, step=0.5)

    run_button = st.button("Run Backtest", type="primary", use_container_width=True)

if run_button:
    config = {
        'min_fvg_size': min_fvg,
        'max_fvg_age_m15': max_fvg_age,
        'rr_target': rr_target,
        'max_risk_pts': max_risk,
        'min_risk_pts': min_risk,
        'max_trades_per_day': max_trades,
        'killzone_start': kz_start,
        'killzone_end': kz_end,
        'use_be': use_be,
        'be_trigger_rr': be_trigger,
        'contract_value': contract_value,
        'target_mode': target_mode,
        'retracement_pct': retracement_pct,
        'cooldown_minutes': cooldown,
        'structure_lookback_days': structure_lookback,
        'entry_start_time': entry_start_time,
        'use_trailing_stop': use_trail,
        'trail_trigger_rr': trail_trigger,
        'trail_offset_pct': trail_offset,
        'use_displacement_filter': use_displacement,
        'min_displacement_body_pct': disp_body_pct,
        'min_displacement_size': disp_size,
        'use_m1_confirmation': use_confirmation,
        'use_session_momentum': use_session_momentum,
        'momentum_threshold': momentum_threshold,
        'use_range_filter': use_range_filter,
        'min_prev_day_range': min_range,
        'max_prev_day_range': max_range,
        'use_structure_confluence': use_confluence,
        'confluence_distance_pts': confluence_dist,
        'use_trend_filter': use_trend,
        'trend_lookback_days': trend_days,
    }

    with st.spinner("Loading data..."):
        raw_df = load_data(selected_files)
        df = get_active_contract(raw_df)

    with st.spinner("Running Multi-TF IFVG backtest..."):
        strategy = IFVGStrategy(config)
        results = strategy.run_backtest(df)
        st.session_state['results'] = results
        st.session_state['config'] = config

if 'results' in st.session_state:
    results = st.session_state['results']
    trades_df = results['trades']
    metrics = results['metrics']

    if trades_df.empty:
        st.warning("No trades were generated. Try adjusting: lower Min FVG Size, increase Max FVG Age, or widen the killzone.")
        st.stop()

    st.header("Performance Overview")

    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    with col1:
        st.metric("Total Trades", metrics['total_trades'])
    with col2:
        st.metric("Win Rate", f"{metrics['win_rate']}%")
    with col3:
        st.metric("Total P&L (pts)", f"{metrics['total_pnl_pts']:+.1f}")
    with col4:
        st.metric("Total P&L ($)", f"${metrics['total_pnl_dollars']:+.2f}")
    with col5:
        st.metric("Profit Factor", f"{metrics['profit_factor']:.2f}")
    with col6:
        st.metric("Trades/Week", f"~{metrics.get('trades_per_week', 0):.1f}")
    with col7:
        st.metric("Trades/Month", f"~{metrics.get('trades_per_month', 0):.0f}")

    col8, col9, col10, col11, col12 = st.columns(5)
    with col8:
        st.metric("Max Drawdown", f"{metrics['max_drawdown_pts']:.1f} pts")
    with col9:
        st.metric("Avg R:R on Wins", f"{metrics['avg_rr_on_wins']:.2f}")
    with col10:
        st.metric("Max Loss Streak", f"{metrics.get('max_consecutive_losses', 0)}")
    with col11:
        st.metric("Win Days", f"{metrics['winning_days']}/{metrics['total_trading_days']}")
    with col12:
        st.metric("Avg Daily P&L", f"{metrics['avg_daily_pnl']:+.1f} pts")

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "Equity Curve", "Trade Log", "Statistics", "Statistical Analysis", "Daily Analysis", "Economic Calendar", "Optimizer", "Strategy Guide"
    ])

    with tab1:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.7, 0.3],
            subplot_titles=("Cumulative P&L (Points)", "Drawdown")
        )

        fig.add_trace(
            go.Scatter(
                x=list(range(len(trades_df))),
                y=trades_df['cum_pnl_pts'],
                mode='lines+markers',
                name='Cumulative P&L',
                line=dict(color='#00d4aa', width=2),
                marker=dict(
                    size=8,
                    color=['#00d4aa' if p > 0 else '#ff4757' for p in trades_df['pnl_pts']],
                ),
                hovertemplate="Trade #%{x}<br>Cum P&L: %{y:.1f} pts<extra></extra>"
            ),
            row=1, col=1,
        )

        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=1)

        fig.add_trace(
            go.Bar(
                x=list(range(len(trades_df))),
                y=trades_df['drawdown'],
                name='Drawdown',
                marker_color='#ff4757',
                opacity=0.6,
            ),
            row=2, col=1,
        )

        fig.update_layout(
            height=550,
            template='plotly_dark',
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=50, r=20, t=40, b=30),
        )
        fig.update_xaxes(title_text="Trade Number", row=2, col=1)
        fig.update_yaxes(title_text="Points", row=1, col=1)
        fig.update_yaxes(title_text="Points", row=2, col=1)

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Equity Over Time")
        fig_time = go.Figure()
        fig_time.add_trace(go.Scatter(
            x=trades_df['entry_time'],
            y=trades_df['cum_pnl_pts'],
            mode='lines+markers',
            name='P&L over time',
            line=dict(color='#00d4aa', width=2),
            marker=dict(
                size=7,
                color=['#00d4aa' if r == 'WIN' else '#ff4757' if r == 'LOSS' else '#ffa502'
                       for r in trades_df['result']],
            ),
        ))
        fig_time.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig_time.update_layout(
            height=350,
            template='plotly_dark',
            xaxis_title="Date",
            yaxis_title="Cumulative P&L (pts)",
            margin=dict(l=50, r=20, t=20, b=30),
        )
        st.plotly_chart(fig_time, use_container_width=True)

    with tab2:
        st.subheader("Trade Log")

        result_filter = st.multiselect(
            "Filter by result",
            options=['WIN', 'LOSS', 'BE', 'EOD'],
            default=['WIN', 'LOSS', 'BE', 'EOD'],
        )
        direction_filter = st.multiselect(
            "Filter by direction",
            options=['BUY', 'SELL'],
            default=['BUY', 'SELL'],
        )

        filtered = trades_df[
            (trades_df['result'].isin(result_filter)) &
            (trades_df['direction'].isin(direction_filter))
        ]

        display_cols = [
            'entry_time', 'direction', 'result', 'entry', 'sl', 'tp',
            'exit_price', 'risk_pts', 'pnl_pts', 'pnl_dollars', 'rr_achieved',
            'fvg_size', 'h1_bias', 'target_mode'
        ]
        available_cols = [c for c in display_cols if c in filtered.columns]

        def color_result(val):
            if val == 'WIN':
                return 'background-color: rgba(0, 212, 170, 0.3)'
            elif val == 'LOSS':
                return 'background-color: rgba(255, 71, 87, 0.3)'
            elif val == 'BE':
                return 'background-color: rgba(255, 165, 2, 0.3)'
            return ''

        def color_pnl(val):
            if isinstance(val, (int, float)):
                if val > 0:
                    return 'color: #00d4aa'
                elif val < 0:
                    return 'color: #ff4757'
            return ''

        styled = filtered[available_cols].style.applymap(
            color_result, subset=['result']
        ).applymap(
            color_pnl, subset=['pnl_pts', 'pnl_dollars']
        )

        st.dataframe(styled, use_container_width=True, height=500)
        st.caption(f"Showing {len(filtered)} of {len(trades_df)} trades")

    with tab3:
        st.subheader("Detailed Statistics")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### Trade Results")
            st.metric("Wins", metrics['wins'])
            st.metric("Losses", metrics['losses'])
            st.metric("Breakevens", metrics['breakevens'])
            st.metric("EOD Exits", metrics['eod_exits'])
            st.metric("Win Rate", f"{metrics['win_rate']}%")

        with col2:
            st.markdown("#### P&L Analysis")
            st.metric("Total P&L (pts)", f"{metrics['total_pnl_pts']:+.2f}")
            st.metric("Avg Win (pts)", f"{metrics['avg_win_pts']:+.2f}")
            st.metric("Avg Loss (pts)", f"{metrics['avg_loss_pts']:.2f}")
            st.metric("Profit Factor", f"{metrics['profit_factor']:.2f}")
            st.metric("Avg R:R on Wins", f"{metrics['avg_rr_on_wins']:.2f}")

        with col3:
            st.markdown("#### Streaks & Risk")
            st.metric("Max Drawdown (pts)", f"{metrics['max_drawdown_pts']:.2f}")
            st.metric("Max Drawdown ($)", f"${metrics['max_drawdown_dollars']:.2f}")
            st.metric("Max Consecutive Wins", metrics['max_consecutive_wins'])
            st.metric("Max Consecutive Losses", metrics['max_consecutive_losses'])

        st.subheader("Result Distribution")
        fig_dist = go.Figure()
        results_counts = trades_df['result'].value_counts()
        colors_map = {'WIN': '#00d4aa', 'LOSS': '#ff4757', 'BE': '#ffa502', 'EOD': '#747d8c'}
        fig_dist.add_trace(go.Pie(
            labels=results_counts.index,
            values=results_counts.values,
            marker_colors=[colors_map.get(r, '#747d8c') for r in results_counts.index],
            hole=0.4,
            textinfo='label+percent+value',
        ))
        fig_dist.update_layout(height=350, template='plotly_dark', margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_dist, use_container_width=True)

        st.subheader("P&L Distribution")
        fig_pnl = go.Figure()
        fig_pnl.add_trace(go.Histogram(x=trades_df['pnl_pts'], nbinsx=20, marker_color='#00d4aa', opacity=0.7))
        fig_pnl.add_vline(x=0, line_dash="dash", line_color="white", opacity=0.5)
        fig_pnl.update_layout(
            height=300, template='plotly_dark',
            xaxis_title="P&L (Points)", yaxis_title="Frequency",
            margin=dict(l=50, r=20, t=20, b=30),
        )
        st.plotly_chart(fig_pnl, use_container_width=True)

        if 'direction' in trades_df.columns:
            st.subheader("Performance by Direction")
            dir_stats = trades_df.groupby('direction').agg(
                count=('pnl_pts', 'count'),
                total_pnl=('pnl_pts', 'sum'),
                avg_pnl=('pnl_pts', 'mean'),
                wins=('result', lambda x: (x == 'WIN').sum()),
            ).reset_index()
            dir_stats['win_rate'] = (dir_stats['wins'] / dir_stats['count'] * 100).round(1)
            st.dataframe(dir_stats, use_container_width=True)

    with tab4:
        st.subheader("Advanced Statistical Analysis")

        stat_results = perform_full_stat_analysis(trades_df)

        cohort = stat_results['cohort']
        robust = stat_results['robustness']
        streaks = stat_results['streaks']
        rr_analysis = stat_results['risk_reward']
        vol = stat_results['volatility']

        st.markdown("### 1. Temporal Cohort Analysis")
        st.caption("Win rate segmented by year, month, and day of week to detect performance drift")

        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            st.metric("Global Win Rate", f"{cohort['global_win_rate']}%")
        with sc2:
            drift = cohort['equity_drift']
            st.metric("Equity Drift (first→last year)", f"{drift:+.1f}%",
                       delta=f"{'Improving' if drift > 0 else 'Declining' if drift < 0 else 'Stable'}")
        with sc3:
            st.metric("Monthly WR Std Dev", f"{cohort['wr_stability_std']:.1f}%",
                       help="Lower = more consistent performance across months")

        yearly_df = pd.DataFrame(cohort['yearly'])
        if not yearly_df.empty:
            fig_yearly = make_subplots(specs=[[{"secondary_y": True}]])
            fig_yearly.add_trace(
                go.Bar(x=yearly_df['year'].astype(str), y=yearly_df['pnl'],
                       name='P&L (pts)', marker_color=['#00d4aa' if p > 0 else '#ff4757' for p in yearly_df['pnl']]),
                secondary_y=False,
            )
            fig_yearly.add_trace(
                go.Scatter(x=yearly_df['year'].astype(str), y=yearly_df['win_rate'],
                           name='Win Rate %', mode='lines+markers',
                           line=dict(color='#ffa502', width=3), marker=dict(size=10)),
                secondary_y=True,
            )
            fig_yearly.update_layout(height=300, template='plotly_dark', title="Yearly Performance",
                                      margin=dict(l=50, r=50, t=40, b=30))
            fig_yearly.update_yaxes(title_text="P&L (pts)", secondary_y=False)
            fig_yearly.update_yaxes(title_text="Win Rate %", secondary_y=True)
            st.plotly_chart(fig_yearly, use_container_width=True)

        monthly_df = pd.DataFrame(cohort['monthly'])
        if not monthly_df.empty:
            fig_monthly = go.Figure()
            fig_monthly.add_trace(go.Bar(
                x=monthly_df['year_month'], y=monthly_df['pnl'],
                name='Monthly P&L',
                marker_color=['#00d4aa' if p > 0 else '#ff4757' for p in monthly_df['pnl']],
            ))
            fig_monthly.add_trace(go.Scatter(
                x=monthly_df['year_month'], y=monthly_df['win_rate'],
                name='Win Rate %', yaxis='y2', mode='lines+markers',
                line=dict(color='#ffa502', width=2), marker=dict(size=5),
            ))
            fig_monthly.update_layout(
                height=350, template='plotly_dark', title="Monthly Breakdown",
                yaxis=dict(title='P&L (pts)'),
                yaxis2=dict(title='Win Rate %', overlaying='y', side='right'),
                margin=dict(l=50, r=50, t=40, b=30),
            )
            st.plotly_chart(fig_monthly, use_container_width=True)

        dow_df = pd.DataFrame(cohort['by_day_of_week'])
        if not dow_df.empty:
            fig_dow = go.Figure()
            fig_dow.add_trace(go.Bar(
                x=dow_df['day_of_week'], y=dow_df['pnl'],
                name='P&L', marker_color=['#00d4aa' if p > 0 else '#ff4757' for p in dow_df['pnl']],
            ))
            fig_dow.add_trace(go.Scatter(
                x=dow_df['day_of_week'], y=dow_df['win_rate'],
                name='Win Rate %', yaxis='y2', mode='lines+markers',
                line=dict(color='#ffa502', width=3), marker=dict(size=10),
            ))
            fig_dow.update_layout(
                height=300, template='plotly_dark', title="Performance by Day of Week",
                yaxis=dict(title='P&L (pts)'),
                yaxis2=dict(title='Win Rate %', overlaying='y', side='right'),
                margin=dict(l=50, r=50, t=40, b=30),
            )
            st.plotly_chart(fig_dow, use_container_width=True)

        st.markdown("---")
        st.markdown("### 2. Statistical Robustness")
        st.caption("Profit factor, Z-Score (randomness test), expectancy, and Kelly criterion")

        rb1, rb2, rb3, rb4 = st.columns(4)
        with rb1:
            st.metric("Profit Factor", f"{robust['profit_factor']:.3f}")
        with rb2:
            st.metric("Expectancy/Trade", f"{robust['expectancy_per_trade']:+.2f} pts")
        with rb3:
            st.metric("Avg Win", f"+{robust['avg_win']:.2f} pts")
        with rb4:
            st.metric("Avg Loss", f"-{robust['avg_loss']:.2f} pts")

        rb5, rb6, rb7, rb8 = st.columns(4)
        with rb5:
            st.metric("Z-Score", f"{robust['z_score']:.3f}",
                       help="Tests if win/loss sequences are random. |Z| > 1.96 = significant pattern")
        with rb6:
            st.metric("P-Value", f"{robust['z_p_value']:.4f}")
        with rb7:
            st.metric("Runs (obs/exp)", f"{robust['runs_count']}/{robust['expected_runs']}")
        with rb8:
            kelly = robust['kelly_fraction']
            st.metric("Kelly Fraction", f"{kelly*100:.1f}%",
                       help="Optimal fraction of capital to risk per trade")

        if robust['z_interpretation']:
            if 'Random' in robust['z_interpretation']:
                st.success(f"Z-Score Interpretation: {robust['z_interpretation']}")
            elif 'Clustering' in robust['z_interpretation']:
                st.warning(f"Z-Score Interpretation: {robust['z_interpretation']}")
            else:
                st.info(f"Z-Score Interpretation: {robust['z_interpretation']}")

        st.markdown("---")
        st.markdown("### 3. Streak Analysis")
        st.caption("Consecutive wins/losses and drawdown recovery")

        sk1, sk2, sk3, sk4 = st.columns(4)
        with sk1:
            st.metric("Max Win Streak", streaks['max_win_streak'])
        with sk2:
            st.metric("Max Loss Streak", streaks['max_loss_streak'])
        with sk3:
            st.metric("Avg Win Streak", f"{streaks['avg_win_streak']:.1f}")
        with sk4:
            st.metric("Avg Loss Streak", f"{streaks['avg_loss_streak']:.1f}")

        sk5, sk6, sk7 = st.columns(3)
        with sk5:
            st.metric("Max Drawdown", f"{streaks['max_drawdown_pts']:.1f} pts")
        with sk6:
            st.metric("DD Duration", f"{streaks['drawdown_duration_trades']} trades")
        with sk7:
            rec = streaks['recovery_trades']
            st.metric("Recovery", f"{rec} trades" if rec is not None else "Not recovered")

        win_dist = streaks['win_streak_distribution']
        loss_dist = streaks['loss_streak_distribution']
        if win_dist or loss_dist:
            sd1, sd2 = st.columns(2)
            with sd1:
                if win_dist:
                    fig_ws = go.Figure(go.Bar(
                        x=[str(k) for k in win_dist.keys()],
                        y=list(win_dist.values()),
                        marker_color='#00d4aa',
                    ))
                    fig_ws.update_layout(height=250, template='plotly_dark',
                                          title="Win Streak Distribution",
                                          xaxis_title="Streak Length", yaxis_title="Count",
                                          margin=dict(l=40, r=20, t=40, b=30))
                    st.plotly_chart(fig_ws, use_container_width=True)
            with sd2:
                if loss_dist:
                    fig_ls = go.Figure(go.Bar(
                        x=[str(k) for k in loss_dist.keys()],
                        y=list(loss_dist.values()),
                        marker_color='#ff4757',
                    ))
                    fig_ls.update_layout(height=250, template='plotly_dark',
                                          title="Loss Streak Distribution",
                                          xaxis_title="Streak Length", yaxis_title="Count",
                                          margin=dict(l=40, r=20, t=40, b=30))
                    st.plotly_chart(fig_ls, use_container_width=True)

        st.markdown("---")
        st.markdown("### 4. Real Risk/Reward Analysis")
        st.caption("Observed win rate vs breakeven win rate, and execution quality")

        rr1, rr2, rr3 = st.columns(3)
        with rr1:
            st.metric("Real R:R Ratio", f"{rr_analysis['real_rr_ratio']:.3f}",
                       help="Average win size / Average loss size")
        with rr2:
            st.metric("Breakeven Win Rate", f"{rr_analysis['breakeven_win_rate']:.1f}%",
                       help="Minimum win rate needed to break even given your R:R")
        with rr3:
            edge = rr_analysis['edge_over_breakeven']
            st.metric("Edge Over Breakeven", f"{edge:+.1f}%",
                       delta=f"{'Positive edge' if edge > 0 else 'Negative edge'}")

        rr4, rr5, rr6 = st.columns(3)
        with rr4:
            st.metric("Avg R:R on Wins", f"{rr_analysis['avg_rr_wins']:.3f}")
        with rr5:
            st.metric("Expectancy (R)", f"{rr_analysis['expectancy_r']:.3f}R",
                       help="Expected return per trade in units of risk")
        with rr6:
            st.metric("Risk Consistency (Std)", f"{rr_analysis['risk_consistency_std']:.2f} pts",
                       help="Lower = more consistent position sizing")

        fig_rr = go.Figure()
        fig_rr.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=rr_analysis['observed_win_rate'],
            title={'text': "Win Rate vs Breakeven"},
            delta={'reference': rr_analysis['breakeven_win_rate'], 'suffix': '%'},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': '#00d4aa'},
                'steps': [
                    {'range': [0, rr_analysis['breakeven_win_rate']], 'color': 'rgba(255,71,87,0.3)'},
                    {'range': [rr_analysis['breakeven_win_rate'], 100], 'color': 'rgba(0,212,170,0.15)'},
                ],
                'threshold': {
                    'line': {'color': '#ffa502', 'width': 4},
                    'thickness': 0.75,
                    'value': rr_analysis['breakeven_win_rate'],
                },
            },
        ))
        fig_rr.update_layout(height=300, template='plotly_dark', margin=dict(l=30, r=30, t=50, b=30))
        st.plotly_chart(fig_rr, use_container_width=True)

        st.markdown("---")
        st.markdown("### 5. Performance Volatility")
        st.caption("Sharpe, Sortino, Calmar ratios and P&L distribution properties")

        vl1, vl2, vl3, vl4 = st.columns(4)
        with vl1:
            st.metric("Sharpe Ratio", f"{vol['sharpe_ratio']:.3f}",
                       help="Mean P&L / Std Dev of P&L per trade")
        with vl2:
            st.metric("Sortino Ratio", f"{vol['sortino_ratio']:.3f}",
                       help="Mean P&L / Downside deviation (penalizes only losses)")
        with vl3:
            st.metric("Calmar Ratio", f"{vol['calmar_ratio']:.3f}",
                       help="Annualized return / Max drawdown")
        with vl4:
            st.metric("Daily Sharpe (ann.)", f"{vol['daily_sharpe_annualized']:.3f}")

        vl5, vl6, vl7 = st.columns(3)
        with vl5:
            st.metric("P&L Std Dev", f"{vol['pnl_std']:.2f} pts")
        with vl6:
            skew = vol['pnl_skew']
            st.metric("Skewness", f"{skew:.3f}",
                       help="Positive = right tail (occasional big wins). Negative = left tail (occasional big losses)")
        with vl7:
            kurt = vol['pnl_kurtosis']
            st.metric("Kurtosis", f"{kurt:.3f}",
                       help="Higher = more extreme outcomes than expected. 0 = normal distribution")

        monthly_rets = vol.get('monthly_returns', [])
        if monthly_rets:
            mr_df = pd.DataFrame(monthly_rets)
            fig_mr = go.Figure()
            fig_mr.add_trace(go.Bar(
                x=mr_df['month'], y=mr_df['pnl'],
                marker_color=['#00d4aa' if p > 0 else '#ff4757' for p in mr_df['pnl']],
            ))
            cum_monthly = mr_df['pnl'].cumsum()
            fig_mr.add_trace(go.Scatter(
                x=mr_df['month'], y=cum_monthly,
                mode='lines', name='Cumulative',
                line=dict(color='#ffa502', width=2),
            ))
            fig_mr.update_layout(
                height=350, template='plotly_dark',
                title="Monthly P&L Distribution",
                xaxis_title="Month", yaxis_title="P&L (pts)",
                margin=dict(l=50, r=20, t=40, b=30),
            )
            st.plotly_chart(fig_mr, use_container_width=True)

            pos_months = sum(1 for r in monthly_rets if r['pnl'] > 0)
            neg_months = sum(1 for r in monthly_rets if r['pnl'] <= 0)
            pnl_values = [r['pnl'] for r in monthly_rets]
            best_month = max(monthly_rets, key=lambda r: r['pnl'])
            worst_month = min(monthly_rets, key=lambda r: r['pnl'])

            vm1, vm2, vm3, vm4 = st.columns(4)
            with vm1:
                st.metric("Positive Months", f"{pos_months}/{len(monthly_rets)}")
            with vm2:
                st.metric("Avg Monthly P&L", f"{np.mean(pnl_values):+.1f} pts")
            with vm3:
                st.metric("Best Month", f"{best_month['month']}: {best_month['pnl']:+.1f}")
            with vm4:
                st.metric("Worst Month", f"{worst_month['month']}: {worst_month['pnl']:+.1f}")

    with tab5:
        st.subheader("Daily Performance Breakdown")

        daily = trades_df.groupby('trade_date').agg(
            trades=('pnl_pts', 'count'),
            pnl=('pnl_pts', 'sum'),
            wins=('result', lambda x: (x == 'WIN').sum()),
        ).reset_index()
        daily['win_rate'] = (daily['wins'] / daily['trades'] * 100).round(1)
        daily['cum_pnl'] = daily['pnl'].cumsum()

        fig_daily = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            row_heights=[0.5, 0.5],
            subplot_titles=("Daily P&L", "Cumulative Daily P&L"),
        )

        fig_daily.add_trace(
            go.Bar(
                x=daily['trade_date'].astype(str),
                y=daily['pnl'],
                name='Daily P&L',
                marker_color=['#00d4aa' if p > 0 else '#ff4757' for p in daily['pnl']],
            ),
            row=1, col=1,
        )

        fig_daily.add_trace(
            go.Scatter(
                x=daily['trade_date'].astype(str),
                y=daily['cum_pnl'],
                mode='lines+markers',
                name='Cumulative',
                line=dict(color='#ffa502', width=2),
            ),
            row=2, col=1,
        )

        fig_daily.update_layout(
            height=550, template='plotly_dark', showlegend=True,
            margin=dict(l=50, r=20, t=40, b=30),
        )
        st.plotly_chart(fig_daily, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Trading Days", metrics['total_trading_days'])
        with col2:
            wd = metrics['winning_days']
            td = max(metrics['total_trading_days'], 1)
            st.metric("Winning Days", f"{wd} ({wd/td*100:.0f}%)")
        with col3:
            st.metric("Avg Daily P&L", f"{metrics['avg_daily_pnl']:+.2f} pts")

        st.dataframe(daily, use_container_width=True)

    with tab6:
        st.subheader("Economic Calendar Impact Analysis")
        st.caption("How does the strategy perform on CPI, PPI, NFP days vs normal days?")

        report = analyze_event_impact(trades_df)
        recommendations = get_event_recommendations(report)

        if not report:
            st.info("No economic event data available for the selected period.")
        else:
            normal_stats = report.get('NORMAL', {})
            if normal_stats:
                st.markdown("#### Baseline: Normal Trading Days")
                nc1, nc2, nc3, nc4 = st.columns(4)
                with nc1:
                    st.metric("Trades", normal_stats.get('trades', 0))
                with nc2:
                    st.metric("Win Rate", f"{normal_stats.get('win_rate', 0)}%")
                with nc3:
                    st.metric("P&L", f"{normal_stats.get('total_pnl', 0):+.1f} pts")
                with nc4:
                    st.metric("Profit Factor", f"{normal_stats.get('profit_factor', 0):.2f}")

            st.markdown("---")
            st.markdown("#### Event Day Analysis")

            for rec in recommendations:
                evt = rec['event']
                stats = rec.get('stats', report.get(evt, {}))

                if rec['verdict'] == 'TRADE':
                    css_class = 'verdict-trade'
                    icon = '✅'
                elif rec['verdict'] == 'AVOID':
                    css_class = 'verdict-avoid'
                    icon = '❌'
                elif rec['verdict'] == 'NEUTRAL':
                    css_class = 'verdict-neutral'
                    icon = '⚠️'
                else:
                    css_class = 'verdict-nodata'
                    icon = '📊'

                with st.expander(f"{icon} {evt} — {rec['verdict']}", expanded=True):
                    st.markdown(f"<span class='{css_class}'>{rec['detail']}</span>", unsafe_allow_html=True)
                    if stats:
                        ec1, ec2, ec3, ec4, ec5 = st.columns(5)
                        with ec1:
                            st.metric("Trades", stats.get('trades', 0))
                        with ec2:
                            st.metric("Win Rate", f"{stats.get('win_rate', 0)}%")
                        with ec3:
                            st.metric("P&L", f"{stats.get('total_pnl', 0):+.1f} pts")
                        with ec4:
                            st.metric("Avg P&L", f"{stats.get('avg_pnl', 0):+.1f} pts")
                        with ec5:
                            st.metric("PF", f"{stats.get('profit_factor', 0):.2f}")

            st.markdown("---")
            st.markdown("#### Full Comparison Table")

            comparison_data = []
            for cat, stats in sorted(report.items()):
                comparison_data.append({
                    'Category': cat,
                    'Trades': stats['trades'],
                    'Days': stats['trading_days'],
                    'Win Rate': f"{stats['win_rate']}%",
                    'Total P&L': f"{stats['total_pnl']:+.1f}",
                    'Avg P&L': f"{stats['avg_pnl']:+.1f}",
                    'PF': f"{stats['profit_factor']:.2f}",
                })
            st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)

            st.markdown("---")
            st.markdown("#### Economic Events Calendar")
            events_df = get_events_df()
            holidays_df = get_holidays_df()

            col_evt, col_hol = st.columns(2)
            with col_evt:
                st.markdown("##### Key Economic Releases")
                st.dataframe(events_df, use_container_width=True)
            with col_hol:
                st.markdown("##### US Market Holidays")
                st.dataframe(holidays_df, use_container_width=True)

    with tab7:
        st.subheader("Parameter Optimizer")
        st.caption("Find the best strategy parameters by testing hundreds of combinations on your data")

        if os.path.exists('optimization_results.json'):
            with open('optimization_results.json', 'r') as f:
                saved_results = json.load(f)

            st.success(f"Last optimization: {saved_results['total_combos_tested']} combinations tested in {saved_results['elapsed_seconds']}s")

            top_saved = pd.DataFrame(saved_results['top_results'])

            st.markdown("#### Top Parameter Sets (ranked by composite score)")
            st.caption("Score combines: P&L, Profit Factor, Calmar Ratio, Win Rate, Trade Count, and Consistency")

            param_cols = ['min_fvg_size', 'max_fvg_age_m15', 'rr_target', 'max_trades_per_day',
                          'retracement_pct', 'min_risk_pts', 'max_risk_pts', 'be_trigger_rr',
                          'trail_trigger_rr', 'trail_offset_pct', 'min_displacement_body_pct',
                          'min_displacement_size', 'entry_start_time', 'cooldown_minutes']
            metric_cols = ['score', 'total_pnl_pts', 'total_pnl_dollars', 'profit_factor', 'win_rate',
                           'max_drawdown_pts', 'total_trades', 'trades_per_week', 'trades_per_month',
                           'calmar_ratio', 'avg_rr_on_wins', 'max_consecutive_losses']

            available_metric_cols = [c for c in metric_cols if c in top_saved.columns]
            available_param_cols = [c for c in param_cols if c in top_saved.columns]

            st.markdown("##### Performance Metrics")
            st.dataframe(
                top_saved[available_metric_cols].head(20).style.background_gradient(
                    subset=['score', 'total_pnl_pts', 'profit_factor'], cmap='RdYlGn'
                ),
                use_container_width=True,
                height=500,
            )

            st.markdown("##### Parameters for Each Set")
            st.dataframe(
                top_saved[available_param_cols].head(20),
                use_container_width=True,
                height=500,
            )

            if len(top_saved) > 0:
                best = top_saved.iloc[0]
                st.markdown("---")
                st.markdown("#### Best Configuration Found")

                bc1, bc2, bc3, bc4, bc5 = st.columns(5)
                with bc1:
                    st.metric("P&L (pts)", f"{best.get('total_pnl_pts', 0):+.1f}")
                with bc2:
                    st.metric("Profit Factor", f"{best.get('profit_factor', 0):.2f}")
                with bc3:
                    st.metric("Win Rate", f"{best.get('win_rate', 0)}%")
                with bc4:
                    st.metric("Max DD", f"{best.get('max_drawdown_pts', 0):.1f}")
                with bc5:
                    st.metric("Calmar", f"{best.get('calmar_ratio', 0):.2f}")

                st.markdown("**Optimal Parameters:**")
                param_display = {}
                for p in param_cols:
                    if p in best:
                        param_display[p] = best[p]
                st.json(param_display)

                if st.button("Apply Best Parameters to Sidebar", key="apply_best"):
                    st.info("Copy these values into the sidebar parameters and re-run the backtest to verify.")

            if len(top_saved) >= 3:
                st.markdown("---")
                st.markdown("#### Parameter Sensitivity Analysis")

                for param in available_param_cols:
                    if top_saved[param].nunique() > 1:
                        fig_sens = go.Figure()
                        fig_sens.add_trace(go.Box(
                            x=top_saved[param].head(20).astype(str),
                            y=top_saved['total_pnl_pts'].head(20),
                            name=param,
                            marker_color='#00d4aa',
                        ))
                        fig_sens.update_layout(
                            height=250,
                            template='plotly_dark',
                            title=f"P&L by {param} (Top 20 configs)",
                            xaxis_title=param,
                            yaxis_title="Total P&L (pts)",
                            margin=dict(l=50, r=20, t=40, b=30),
                        )
                        st.plotly_chart(fig_sens, use_container_width=True)
        else:
            st.info("No optimization results found yet.")

        st.markdown("---")
        st.markdown("#### Run New Optimization")

        grid_size = get_param_grid_size()
        oc1, oc2 = st.columns(2)
        with oc1:
            n_combos = st.slider("Number of random combinations to test", 50, min(500, grid_size),
                                  min(100, grid_size), 50,
                                  help=f"Full grid has {grid_size:,} combinations. ~0.6s per combo. Random sampling finds good solutions efficiently.")
        with oc2:
            est_time = n_combos * 0.65
            if est_time < 60:
                st.metric("Estimated Time", f"~{max(1, int(est_time))} seconds")
            else:
                st.metric("Estimated Time", f"~{est_time/60:.1f} minutes")

        if st.button("Start Optimization", type="primary", use_container_width=True):
            progress_bar = st.progress(0, text="Loading data...")
            status_text = st.empty()

            raw_df = load_data(selected_files)
            df_opt = get_active_contract(raw_df)

            def update_progress(done, total, elapsed, eta, msg=None):
                pct = done / total if total > 0 else 0
                if msg:
                    progress_bar.progress(pct, text=msg)
                else:
                    progress_bar.progress(pct, text=f"Testing {done}/{total} | Elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s")

            opt_result = run_optimization(
                df_opt,
                param_grid=PARAM_GRID,
                fixed_params=FIXED_PARAMS,
                top_n=30,
                max_combos=n_combos,
                progress_callback=update_progress,
            )

            progress_bar.progress(1.0, text="Done!")

            top_results = opt_result['top_n'].head(30).to_dict('records')
            for r in top_results:
                for k, v in r.items():
                    if hasattr(v, 'item'):
                        r[k] = v.item()

            with open('optimization_results.json', 'w') as f:
                json.dump({
                    'total_combos_tested': opt_result['total_combos'],
                    'elapsed_seconds': opt_result['elapsed'],
                    'top_results': top_results,
                }, f, indent=2, default=str)

            st.success(f"Optimization complete! Tested {opt_result['total_combos']} combinations in {opt_result['elapsed']}s")
            st.rerun()

    with tab8:
        st.subheader("Multi-TF IFVG Strategy — Guide Complet de Prise de Position")
        st.caption("Toutes les phases, analyses et conditions requises, de la preparation pre-session jusqu'a la gestion active du trade")

        config_used = st.session_state.get('config', {})

        st.markdown("---")
        st.markdown("## Phase 0 : Preparation Pre-Session")
        st.markdown("""
**Objectif** : Rassembler toutes les donnees de contexte AVANT l'ouverture de la killzone.

**Donnees requises** :
- Flux de prix M1 (1 minute) sur MNQ en temps reel, fuseau horaire **Eastern Time (ET)**
- Historique OHLCV (Open, High, Low, Close, Volume) sur les **20 derniers jours de trading** (configurable)

**Niveaux de structure a calculer** :
1. **Daily Highs / Lows** : Le high et le low de chacun des 20 derniers jours
2. **Weekly Highs / Lows** : Le high/low de chaque semaine glissante (blocs de 5 jours)
3. **Swing Highs / Swing Lows** : Points pivots ou un high depasse les highs adjacents (jour precedent ET jour suivant) — idem pour les lows
4. **PDH / PDL** : Les niveaux daily high/low les plus proches du prix actuel (apres tri par proximite), servant de reference immediate

Seuls les niveaux situes a **moins de 500 points du prix actuel** sont retenus, tries par proximite (les 10 plus proches par categorie). PDH et PDL sont les premiers elements de ces listes triees.
""")

        st.markdown("---")
        st.markdown("## Phase 1 : Filtres Journaliers (Pre-Killzone)")
        st.markdown("""
**Objectif** : Eliminer les jours a faible potentiel AVANT toute analyse.

**1a. Filtre par jour de la semaine** *(optionnel)*
- Si active : seuls les jours autorises sont trades (ex: lundi a vendredi, ou un sous-ensemble)
- Certains jours (ex: vendredi) peuvent historiquement sous-performer

**1b. Filtre de range du jour precedent** *(optionnel)*
- Calcul : `range_veille = high_veille - low_veille`
- Le range doit etre compris entre un minimum et un maximum (defaut: 60 a 400 points)
- **Trop petit** (<60 pts) = marche trop calme, pas assez de volatilite pour des mouvements propres
- **Trop grand** (>400 pts) = marche trop erratique, risque eleve de faux signaux
""")

        st.markdown("---")
        st.markdown("## Phase 2 : Determination du Biais H1 (Horaire)")
        st.markdown("""
**Objectif** : Definir la direction dominante du marche AVANT la killzone. C'est le filtre directionnel principal.

**Methode** :
1. Prendre les donnees pre-killzone (tout ce qui precede 09:30 ET)
2. Re-echantillonner en bougies H1 (1 heure)
3. Analyser la **derniere bougie H1 completee** :
   - Si **close > open** → Biais **ACHAT (BUY)**
   - Si **close < open** → Biais **VENTE (SELL)**

**Fallback** : Si moins de 60 minutes de donnees pre-KZ sont disponibles, utiliser la session 08:30-09:30 ET :
- Premier open vs dernier close de cette periode

**Regle absolue** : Sans biais H1 determine, **aucun trade n'est pris ce jour-la**.

**Filtre additionnel — Tendance multi-jours** *(optionnel)* :
- Analyse des N derniers jours (defaut: 3 jours)
- Score haussier = jours haussiers + higher highs + mouvement net positif
- Score baissier = jours baissiers + lower lows + mouvement net negatif
- Si l'ecart entre scores est >= 2, une tendance est identifiee
- **Le biais multi-jours doit concorder avec le biais H1**, sinon on ne trade pas
""")

        st.markdown("---")
        st.markdown("## Phase 3 : Momentum Pre-Session *(optionnel)*")
        st.markdown("""
**Objectif** : Verifier que l'elan pre-killzone ne contredit pas le biais H1.

**Methode** :
1. Analyser la session 08:00-09:30 ET (pre-marche)
2. Calculer : `momentum = (dernier_close - premier_open) / (high_max - low_min)`
3. Le momentum est un ratio entre -1 et +1

**Conditions d'exclusion** :
- Biais **BUY** mais momentum < -0.4 (forte pression vendeuse pre-session) → **Pas de trade**
- Biais **SELL** mais momentum > +0.4 (forte pression acheteuse pre-session) → **Pas de trade**

La logique : si les institutionnels poussent fort dans la direction opposee au biais avant l'ouverture, le biais H1 est probablement invalide.
""")

        st.markdown("---")
        st.markdown("## Phase 4 : Detection des FVG sur M15")
        st.markdown("""
**Objectif** : Identifier les desequilibres de prix (Fair Value Gaps) sur le timeframe 15 minutes.

**Construction des bougies M15** :
- Re-echantillonner les donnees M1 (killzone + pre-killzone) en bougies de 15 minutes
- Minimum 4 bougies M15 requises pour l'analyse

**Definition d'un FVG** (3 bougies consecutives) :

**FVG Haussier (Bullish)** :
```
Bougie 1 (high)  ─────────
                     GAP ↑    ← Ce GAP est le FVG
Bougie 3 (low)   ─────────
```
- Condition : `low_bougie3 - high_bougie1 >= taille_minimum` (defaut: 3.0 pts)
- La bougie du milieu (bougie 2) doit etre **haussiere** (close > open)
- Zone du FVG : de `high_bougie1` (bottom) a `low_bougie3` (top)

**FVG Baissier (Bearish)** :
```
Bougie 3 (high)  ─────────
                     GAP ↓    ← Ce GAP est le FVG
Bougie 1 (low)   ─────────
```
- Condition : `low_bougie1 - high_bougie3 >= taille_minimum` (defaut: 3.0 pts)
- La bougie du milieu (bougie 2) doit etre **baissiere** (close < open)
- Zone du FVG : de `high_bougie3` (bottom) a `low_bougie1` (top)

**Proprietes enregistrees** : top, bottom, midpoint (point milieu), taille en points, timestamp
""")

        st.markdown("---")
        st.markdown("## Phase 5 : Filtre de Displacement (Qualite du FVG)")
        st.markdown("""
**Objectif** : Ne garder que les FVG crees par un mouvement **impulsif et decisif** — signe d'activite institutionnelle.

**Analyse de la bougie du milieu** (la bougie de displacement, bougie 2 du pattern) :

**Critere 1 — Body Ratio** :
```
body_ratio = |close - open| / (high - low)
```
- Doit etre >= **55%** (defaut)
- Signifie que le corps represente au moins 55% du range total
- Filtre les bougies avec trop de meches (indecision)

**Critere 2 — Taille minimum du body** :
```
body_size = |close - open|
```
- Doit etre >= **3.5 points** (defaut)
- Filtre les micro-mouvements non significatifs

**Les deux criteres doivent etre remplis** simultanement. Un FVG cree par une bougie a petits corps ou a grandes meches est considere comme faible et est elimine.
""")

        st.markdown("---")
        st.markdown("## Phase 6 : Detection de l'Inversion (IFVG)")
        st.markdown("""
**Objectif** : Identifier le moment ou un FVG est **inverse** — c'est-a-dire que le prix traverse completement le FVG et cloture de l'autre cote. C'est le signal principal de la strategie.

**Logique d'inversion** :

**FVG Haussier → Signal SELL** (uniquement si biais H1 = SELL) :
- On surveille les barres M15 suivant la creation du FVG
- Si le **close** d'une bougie M15 descend **en dessous du bottom du FVG** → le FVG est inverse
- Le FVG haussier est "viole" par le bas = les acheteurs ont ete pieges = signal de vente
- **Zone de trade SELL** : du **top du FVG** (haut) au **midpoint** (milieu)

**FVG Baissier → Signal BUY** (uniquement si biais H1 = BUY) :
- Si le **close** d'une bougie M15 monte **au-dessus du top du FVG** → le FVG est inverse
- Le FVG baissier est "viole" par le haut = les vendeurs ont ete pieges = signal d'achat
- **Zone de trade BUY** : du **midpoint** (milieu) au **bottom du FVG** (bas)

**Contrainte temporelle** : L'inversion doit se produire dans les **15 barres M15** (3h45) suivant la creation du FVG. Au-dela, le FVG est considere comme expire.

**Alignement obligatoire** : L'inversion ne produit un signal que si elle est **dans la direction du biais H1**.
""")

        st.markdown("---")
        st.markdown("## Phase 7 : Filtres Additionnels sur les IFVGs *(optionnels)*")
        st.markdown("""
**7a. Confluence Structurelle** :
- Le milieu de la zone IFVG doit etre a **moins de X points** (defaut: 50 pts) d'un niveau de structure cle
- Niveaux consideres : daily highs/lows, swing highs/lows, weekly highs/lows, PDH/PDL
- Plus un IFVG est proche d'un niveau de structure, plus il a de probabilite de provoquer une reaction

**7b. Liquidity Sweep** :
- Avant ou au moment de l'inversion, le prix doit avoir "balaye" (sweep) un recent high ou low
- Le lookback pour les highs/lows recents est de **20 barres M15** (configurable via `sweep_lookback_bars`)
- Pour un signal SELL : le high des 3 barres post-inversion doit approcher le recent high (a 0.1% pres)
- Pour un signal BUY : le low des 3 barres post-inversion doit approcher le recent low (a 0.1% pres)
- La logique ICT : les smart money provoquent un sweep de liquidite avant de reverser
""")

        st.markdown("---")
        st.markdown("## Phase 8 : Recherche d'Entree sur M1 (Killzone)")
        st.markdown("""
**Objectif** : Trouver le point d'entree precis sur le timeframe 1 minute, en attendant un retracement dans la zone IFVG.

**Fenetre d'entree** :
- **Debut** : 09:45 ET (par defaut, retarde de 15 min apres l'ouverture pour filtrer la volatilite initiale)
- **Fin** : 11:00 ET (fin de killzone)
- Le retracement doit se produire **APRES** l'inversion (pas avant)

**Condition de retracement** (barre par barre sur M1) :

**Pour un signal SELL** :
```
La bougie M1 doit monter jusque dans la zone IFVG :
- high >= zone_top - (zone_range x 60%)     ← penetration d'au moins 60% de la zone
- close < zone_top                            ← le close reste sous le haut de la zone
```
Le prix retrace vers le haut dans la zone de vente, mais ne la depasse pas — signe que les vendeurs reprennent le controle.

**Pour un signal BUY** :
```
La bougie M1 doit descendre jusque dans la zone IFVG :
- low <= zone_bottom + (zone_range x 60%)   ← penetration d'au moins 60% de la zone
- close > zone_bottom                        ← le close reste au-dessus du bas de la zone
```
Le prix retrace vers le bas dans la zone d'achat, mais ne la casse pas — signe que les acheteurs reprennent le controle.

**Cooldown** : Minimum **10 minutes** entre deux entrees (evite le sur-trading apres un stop).
""")

        st.markdown("---")
        st.markdown("## Phase 9 : Confirmation M1 (Bougie de Validation)")
        st.markdown("""
**Objectif** : Exiger une bougie de confirmation sur M1 pour valider l'entree — ultime filtre de qualite.

**Pour un signal SELL** — La bougie M1 d'entree doit etre :
1. **Baissiere** : close < open
2. **Corps significatif** : `|close - open| / (high - low) > 30%`
3. **ET** au moins un de ces criteres :
   - **Meche haute** (upper wick) > 30% de la taille du corps → rejet visible du haut
   - **Body dominant** : corps > 50% du range total → mouvement decisif

**Pour un signal BUY** — La bougie M1 d'entree doit etre :
1. **Haussiere** : close > open
2. **Corps significatif** : `(close - open) / (high - low) > 30%`
3. **ET** au moins un de ces criteres :
   - **Meche basse** (lower wick) > 30% de la taille du corps → rejet visible du bas
   - **Body dominant** : corps > 50% du range total → mouvement decisif

**Interpretation** : La bougie de confirmation montre que le prix a ete rejete dans la zone IFVG. C'est la preuve visuelle que les participants ont reagi au niveau.
""")

        st.markdown("---")
        st.markdown("## Phase 9b : Momentum M1 *(optionnel)*")
        st.markdown("""
**Objectif** : Verifier que les dernieres bougies M1 montrent un elan coherent avec la direction du trade.

**Methode** :
1. Analyser les **5 dernieres bougies M1** avant l'entree (configurable via `momentum_bars`)
2. Calculer un score :
   - +1 pour chaque bougie dont le corps est dans le sens du trade (haussiere pour BUY, baissiere pour SELL)
   - +1 si le mouvement net des 5 bougies est dans le sens du trade
3. Le score doit atteindre au minimum **3** (configurable via `momentum_min_score`)

**Si active** : Ce filtre s'applique **apres** la confirmation M1. Les deux doivent etre valides pour entrer.
""")

        st.markdown("---")
        st.markdown("## Phase 10 : Execution du Trade")
        st.markdown("""
**Objectif** : Definir prix d'entree, stop loss, et take profit avec precision.

**Prix d'entree** : **Close de la bougie M1 de confirmation**

**Stop Loss** :
| Direction | Placement du SL |
|-----------|----------------|
| **SELL** | `zone_top + 2.0 points` (au-dessus du haut de la zone IFVG + buffer) |
| **BUY** | `zone_bottom - 2.0 points` (en dessous du bas de la zone IFVG + buffer) |

Le buffer de 2 points evite les sorties sur simple bruit de marche (meches qui touchent exactement le niveau).

**Calcul du risque** :
```
risk = |entry_price - stop_loss|
```
- Le risque doit etre entre **5.0 et 25.0 points** (defaut)
- Si le risque est hors de cette fourchette, le trade est **annule**
- Trop petit (<5 pts) = stop trop serre, arrete par le bruit
- Trop grand (>25 pts) = risque disproportionne

**Take Profit** :

| Mode | Calcul du TP |
|------|-------------|
| **Fixed R:R** | `entry +/- (risk x 1.2)` (defaut R:R = 1.2) |
| **SSL/BSL** | Niveau de liquidite le plus proche a minimum 1.5x le risque de distance |

**Mode SSL/BSL (Liquidity)** :
- SELL : cherche le swing low ou daily low le plus proche EN DESSOUS de l'entree, a au moins 1.5x risk de distance
- BUY : cherche le swing high ou daily high le plus proche AU DESSUS de l'entree, a au moins 1.5x risk de distance
- Si aucun niveau n'est trouve, fallback sur le Fixed R:R

**Valeur du point** : $2.00 par point (specification MNQ)
""")

        st.markdown("---")
        st.markdown("## Phase 11 : Gestion Active du Trade")
        st.markdown("""
**Objectif** : Proteger le capital et securiser les profits en cours de trade.

**11a. Breakeven (Protection du capital)** :
- **Declenchement** : Quand le prix bouge de **0.5R** en faveur du trade
  - BUY : high atteint `entry + (risk x 0.5)`
  - SELL : low atteint `entry - (risk x 0.5)`
- **Action** : Le stop loss est deplace a `entry +/- 1 point` (breakeven + 1 pt pour couvrir les frais)
- **Important** : Le breakeven ne s'active que si le **trailing stop n'est pas deja actif**. Si le trailing est declenche en premier, c'est lui qui gere le SL.
- Desormais, le trade ne peut plus etre perdant (sauf gap)

**11b. Trailing Stop (Securisation des profits)** :
- **Declenchement** : Quand le prix bouge de **0.5R** en faveur (meme seuil que BE)
- **Fonctionnement** :
  - Le SL suit le meilleur prix atteint, a une distance de **30% du risque initial**
  - BUY : `trailing_SL = best_high - (risk x 30%)`
  - SELL : `trailing_SL = best_low + (risk x 30%)`
  - Le trailing SL ne peut que se rapprocher du prix (jamais reculer)
- **Interaction BE / Trailing** : Le trailing stop a priorite sur le breakeven. Si les deux ont le meme seuil de declenchement (0.5R), le trailing prend le relais et le BE ne s'applique plus. Le SL final est toujours le plus favorable des deux.

**11c. Sortie de fin de journee (EOD)** :
- Si ni le TP ni le SL n'est touche avant la fin de la session
- Le trade est ferme au **dernier close disponible**
- Resultat marque comme "EOD" (End Of Day)

**Priorite de sortie** (evaluee barre par barre) :
1. Stop Loss touche → sortie immediate
2. Take Profit touche → sortie immediate
3. Mise a jour trailing stop → ajustement du SL
4. Activation breakeven → ajustement du SL
5. Fin de journee → sortie forcee
""")

        st.markdown("---")
        st.markdown("## Phase 12 : Limites de Position et Regles de Gestion")
        st.markdown("""
**Objectif** : Controler l'exposition et eviter le sur-trading.

| Regle | Valeur par defaut |
|-------|------------------|
| Max trades par jour | **2** |
| Cooldown entre trades | **10 minutes** |
| Chaque IFVG utilise une seule fois | Oui |
| Killzone | **09:30 - 11:00 ET** |
| Entrees a partir de | **09:45 ET** |

**Logique du cooldown** : Apres chaque trade (gagnant ou perdant), attendre au minimum 10 minutes avant de chercher une nouvelle entree. Empeche les entrees emotionnelles consecutives.

**Utilisation unique des IFVG** : Une fois qu'un IFVG a genere un trade (gagnant ou perdant), il ne peut plus etre utilise. Cela evite de re-entrer sur une zone deja exploitee.
""")

        st.markdown("---")
        st.markdown("## Resume Visuel : Checklist de Prise de Position")

        st.markdown("""
```
AVANT LA SESSION (Pre-09:30 ET)
================================
[1] Calculer les niveaux de structure (daily/weekly/swing H/L, PDH/PDL)
[2] Verifier filtre jour de la semaine (si active)
[3] Verifier range du jour precedent (si active)
[4] Determiner le biais H1 (derniere bougie horaire avant 09:30)
[5] Verifier concordance avec tendance multi-jours (si active)
[6] Verifier momentum pre-session 08:00-09:30 (si active)

CONSTRUCTION DU SIGNAL (M15)
================================
[7]  Construire les bougies M15 (killzone + pre-killzone)
[8]  Detecter tous les FVG M15 (gap >= 3.0 pts, bougie milieu directionnelle)
[9]  Filtrer par displacement (body >= 55%, taille >= 3.5 pts)
[10] Detecter les inversions (close au-dela du FVG, dans le sens du biais H1)
[11] Appliquer filtres optionnels (confluence, liquidity sweep)

ENTREE (M1 — a partir de 09:45 ET)
================================
[12] Scanner chaque bougie M1 dans la killzone
[13] Verifier retracement dans la zone IFVG (penetration >= 60%)
[14] Valider la bougie de confirmation M1 (corps significatif + rejet)
[15] Calculer entry/SL/TP et verifier que le risque est entre 5-25 pts

GESTION DU TRADE
================================
[16] Surveiller barre par barre : SL → TP → trailing → breakeven → EOD
[17] Respecter max 2 trades/jour et cooldown 10 min
```
""")

        st.markdown("---")
        st.markdown("## Parametres Optimises (Configuration Actuelle)")
        if config_used:
            param_display = {
                'Taille min FVG': f"{config_used.get('min_fvg_size', 3.0)} pts",
                'Age max FVG (barres M15)': f"{config_used.get('max_fvg_age_m15', 15)}",
                'R:R cible': f"{config_used.get('rr_target', 1.2)}",
                'Risque max': f"{config_used.get('max_risk_pts', 25.0)} pts",
                'Risque min': f"{config_used.get('min_risk_pts', 5.0)} pts",
                'Max trades/jour': f"{config_used.get('max_trades_per_day', 2)}",
                'Killzone': f"{config_used.get('killzone_start', '09:30')} - {config_used.get('killzone_end', '11:00')} ET",
                'Debut des entrees': f"{config_used.get('entry_start_time', '09:45')} ET",
                'Cooldown': f"{config_used.get('cooldown_minutes', 10)} min",
                'Retracement': f"{config_used.get('retracement_pct', 60)}%",
                'Breakeven trigger': f"{config_used.get('be_trigger_rr', 0.5)}R",
                'Trailing trigger': f"{config_used.get('trail_trigger_rr', 0.5)}R",
                'Trailing offset': f"{config_used.get('trail_offset_pct', 30)}%",
                'Displacement body min': f"{config_used.get('min_displacement_body_pct', 55)}%",
                'Displacement taille min': f"{config_used.get('min_displacement_size', 3.5)} pts",
                'Mode target': config_used.get('target_mode', 'fixed_rr'),
                'Valeur du point': f"${config_used.get('contract_value', 2.0):.2f}",
            }
            pc1, pc2 = st.columns(2)
            items = list(param_display.items())
            mid = len(items) // 2
            with pc1:
                for k, v in items[:mid]:
                    st.markdown(f"**{k}** : `{v}`")
            with pc2:
                for k, v in items[mid:]:
                    st.markdown(f"**{k}** : `{v}`")
        else:
            st.info("Lancez un backtest pour voir les parametres utilises.")

else:
    st.info("Configure strategy parameters in the sidebar and click **Run Backtest** to start.")

    with st.expander("About the Multi-TF IFVG Strategy"):
        st.markdown("""
        **Multi-Timeframe Inversed Fair Value Gap (IFVG)** approach:

        1. **H1 Bias** — Determine the hourly trend direction before the killzone opens

        2. **Market Structure** — Compute daily, weekly and swing highs/lows as liquidity targets and reference points

        3. **M15 FVG Detection** — Identify Fair Value Gaps on the 15-minute timeframe
           - Bullish FVG: Gap between candle 1 high and candle 3 low
           - Bearish FVG: Gap between candle 1 low and candle 3 high

        4. **M15 Inversion** — Watch for price to trade through and close beyond the FVG (aligned with H1 bias)

        5. **M1 Retracement Entry** — Wait for price to retrace back into the inverted FVG zone on the 1-minute chart

        6. **Target Options**:
           - **Fixed R:R** — Take profit at a fixed multiple of risk
           - **SSL/BSL** — Target the nearest sell-side or buy-side liquidity level

        7. **Killzone** — All entries between 9:30-11:00 ET only
        """)

    if data_files:
        st.markdown(f"**{len(data_files)} data file(s) available:** {', '.join(file_labels)}")
