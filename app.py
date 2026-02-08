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

    min_fvg = st.slider("Min FVG Size (pts)", 1.0, 15.0, 4.0, 0.5,
                         help="Minimum gap size on M15 to qualify as FVG")
    max_fvg_age = st.slider("Max FVG Age (M15 bars)", 2, 24, 12, 1,
                             help="Maximum age of M15 FVG before expiry")
    rr_target = st.slider("Risk:Reward Target", 1.0, 5.0, 3.0, 0.5,
                           help="Target R:R ratio for take profit")
    max_risk = st.slider("Max Risk (pts)", 10.0, 100.0, 50.0, 5.0)
    min_risk = st.slider("Min Risk (pts)", 1.0, 20.0, 5.0, 1.0)
    max_trades = st.slider("Max Trades / Day", 1, 4, 2, 1)
    retracement_pct = st.slider("Retracement % into IFVG zone", 20, 80, 50, 5,
                                 help="How deep price must retrace into the inverted FVG zone on M1")
    cooldown = st.slider("Cooldown (minutes)", 0, 30, 10, 1)
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
    be_trigger = st.slider("BE Trigger (xR)", 0.5, 3.0, 1.5, 0.1) if use_be else 1.5
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

    col1, col2, col3, col4, col5, col6 = st.columns(6)
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
        st.metric("Trades/Month", f"~{metrics.get('trades_per_month', 0):.0f}")

    col7, col8, col9, col10 = st.columns(4)
    with col7:
        st.metric("Max Drawdown", f"{metrics['max_drawdown_pts']:.1f} pts")
    with col8:
        st.metric("Avg R:R on Wins", f"{metrics['avg_rr_on_wins']:.2f}")
    with col9:
        st.metric("Win Days", f"{metrics['winning_days']}/{metrics['total_trading_days']}")
    with col10:
        st.metric("Avg Daily P&L", f"{metrics['avg_daily_pnl']:+.1f} pts")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Equity Curve", "Trade Log", "Statistics", "Daily Analysis", "Economic Calendar", "Optimizer"
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

    with tab5:
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

    with tab6:
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
                          'retracement_pct', 'min_risk_pts', 'max_risk_pts', 'be_trigger_rr', 'cooldown_minutes']
            metric_cols = ['score', 'total_pnl_pts', 'total_pnl_dollars', 'profit_factor', 'win_rate',
                           'max_drawdown_pts', 'total_trades', 'trades_per_month', 'calmar_ratio',
                           'avg_rr_on_wins', 'max_consecutive_losses']

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
            n_combos = st.slider("Number of random combinations to test", 50, min(2000, grid_size),
                                  min(500, grid_size), 50,
                                  help=f"Full grid has {grid_size:,} combinations. Random sampling finds good solutions efficiently.")
        with oc2:
            est_time = n_combos * 0.015
            st.metric("Estimated Time", f"~{max(1, int(est_time))} seconds")

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
