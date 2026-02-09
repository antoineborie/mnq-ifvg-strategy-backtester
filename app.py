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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #161b22 0%, #1c2333 100%);
        border: 1px solid rgba(0, 212, 170, 0.12);
        border-radius: 12px;
        padding: 16px 18px;
        transition: border-color 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        border-color: rgba(0, 212, 170, 0.35);
    }
    div[data-testid="stMetric"] label {
        font-size: 0.72rem;
        font-weight: 500;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: #8b949e;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-weight: 600;
        font-size: 1.3rem;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: #161b22;
        border-radius: 10px;
        padding: 4px;
        border: 1px solid #21262d;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
        font-size: 0.82rem;
        font-weight: 500;
        letter-spacing: 0.01em;
        color: #8b949e;
        border: none;
        background: transparent;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #c9d1d9;
        background: rgba(0, 212, 170, 0.06);
    }
    .stTabs [aria-selected="true"] {
        background: rgba(0, 212, 170, 0.12) !important;
        color: #00d4aa !important;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        display: none;
    }
    .stTabs [data-baseweb="tab-border"] {
        display: none;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #21262d;
        border-radius: 10px;
        overflow: hidden;
    }

    div[data-testid="stExpander"] {
        border: 1px solid #21262d;
        border-radius: 10px;
        background: #161b22;
    }
    div[data-testid="stExpander"] summary {
        font-weight: 500;
    }

    .main-header {
        padding: 1.2rem 0 0.6rem 0;
    }
    .main-header h1 {
        font-size: 1.75rem;
        font-weight: 700;
        color: #f0f6fc;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .main-header p {
        font-size: 0.85rem;
        color: #8b949e;
        margin: 4px 0 0 0;
        font-weight: 400;
    }

    .section-divider {
        border: none;
        border-top: 1px solid #21262d;
        margin: 1.5rem 0;
    }

    h2, h3 {
        letter-spacing: -0.01em;
    }

    .verdict-trade { color: #00d4aa; font-weight: 600; font-size: 16px; }
    .verdict-avoid { color: #ff4757; font-weight: 600; font-size: 16px; }
    .verdict-neutral { color: #ffa502; font-weight: 600; font-size: 16px; }
    .verdict-nodata { color: #8b949e; font-size: 14px; }

    button[kind="primary"] {
        border-radius: 8px;
        font-weight: 600;
        letter-spacing: 0.02em;
    }

    .stSlider label, .stCheckbox label, .stSelectbox label, .stMultiSelect label {
        font-size: 0.82rem;
        font-weight: 500;
    }

    section[data-testid="stSidebar"] {
        background: #0d1117;
        border-right: 1px solid #21262d;
    }
    section[data-testid="stSidebar"] .stMarkdown h2 {
        font-size: 1rem;
        font-weight: 600;
        color: #f0f6fc;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        padding-bottom: 4px;
        border-bottom: 1px solid #21262d;
        margin-bottom: 12px;
    }
    section[data-testid="stSidebar"] .stMarkdown h3 {
        font-size: 0.8rem;
        font-weight: 600;
        color: #00d4aa;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-top: 16px;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>MNQ Futures — IFVG Strategy Backtester</h1>
    <p>Multi-Timeframe Inversed Fair Value Gap &nbsp;|&nbsp; H1 Bias → M15 IFVG → M1 Retracement Entry</p>
</div>
""", unsafe_allow_html=True)

CHART_BASE = dict(
    template='plotly_dark',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(22,27,34,0.6)',
    font=dict(family='Inter, -apple-system, sans-serif', color='#c9d1d9', size=12),
    margin=dict(l=50, r=20, t=40, b=30),
    legend=dict(font=dict(size=11)),
)
AXIS_DEFAULTS = dict(gridcolor='#21262d', zerolinecolor='#30363d')
CHART_LAYOUT = {**CHART_BASE, 'xaxis': dict(**AXIS_DEFAULTS), 'yaxis': dict(**AXIS_DEFAULTS)}

COLORS = {
    'green': '#00d4aa',
    'red': '#ff4757',
    'orange': '#ffa502',
    'blue': '#58a6ff',
    'gray': '#8b949e',
    'line': '#ffa502',
}

data_files = list_data_files()
data_files_2024_plus = [f for f in data_files if any(y in os.path.basename(f) for y in ['2024', '2025', '2026'])]
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
        default=data_files_2024_plus,
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
    rr_target = st.slider("Risk:Reward Target", 0.5, 5.0, 0.8, 0.1,
                           help="Target R:R ratio for take profit")
    max_risk = st.slider("Max Risk (pts)", 5.0, 60.0, 25.0, 1.0)
    min_risk = st.slider("Min Risk (pts)", 1.0, 15.0, 5.0, 1.0)
    max_trades = st.slider("Max Trades / Day", 1, 4, 1, 1)
    retracement_pct = st.slider("Retracement % into IFVG zone", 20, 80, 60, 5,
                                 help="How deep price must retrace into the inverted FVG zone on M1")
    cooldown = st.slider("Cooldown (minutes)", 0, 30, 10, 1)
    entry_start_time = st.selectbox("Entry Start Time (ET)", ['09:30', '09:35', '09:40', '09:45', '09:50', '09:55', '10:00', '10:05', '10:10', '10:15'],
                                     index=6, help="Only look for entries after this time (later = higher win rate)")
    structure_lookback = st.slider("Structure Lookback (days)", 5, 60, 20, 5,
                                    help="Days of history for daily/weekly structure levels")

    st.subheader("Killzone (ET)")
    kz_start = st.text_input("Start", "09:30")
    kz_end = st.text_input("End", "12:00")

    st.subheader("Target Mode")
    target_mode = st.radio("Take Profit Method", ["fixed_rr", "ssl"],
                            format_func=lambda x: "Fixed R:R" if x == "fixed_rr" else "SSL/BSL (Liquidity Levels)",
                            help="Fixed R:R uses your R:R target. SSL targets nearest liquidity level.")

    st.subheader("Risk Management")
    use_be = st.checkbox("Breakeven Protection", value=True)
    be_trigger = st.slider("BE Trigger (xR)", 0.3, 2.0, 0.5, 0.1,
                            help="Move stop to breakeven after price moves this many R in your favor") if use_be else 0.5
    use_stop_after_loss = st.checkbox("Stop Apres Perte", value=True,
                                       help="Arreter de trader pour la journee apres la premiere perte")

    partial_tp_pct = 100

    st.subheader("Trailing Stop")
    use_trail = st.checkbox("Trailing Stop", value=True)
    if use_trail:
        trail_trigger = st.slider("Trail Trigger (xR)", 0.3, 2.0, 0.3, 0.1,
                                   help="Start trailing after this many R in profit")
        trail_offset = st.slider("Trail Offset (%)", 10, 70, 25, 5,
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
    use_opening_range_filter = st.checkbox("Filtre Opening Range", value=False,
                                            help="L'Opening Range 09:30-09:45 doit confirmer le biais H1")

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

    st.subheader("Volatility Regime")
    use_vol_regime = st.checkbox("Adaptive Volatility Filter", value=False,
                                  help="Automatically adjust parameters based on market volatility (ATR). Reduces R:R and tightens filters during low-vol periods for more consistent monthly performance.")
    if use_vol_regime:
        vol_atr_period = st.slider("ATR Period (days)", 5, 20, 10, 1,
                                    help="Number of days to compute average daily range")
        vol_low_pct = st.slider("Low Vol Percentile", 10, 40, 30, 5,
                                 help="Below this percentile = low volatility regime")
        vol_high_pct = st.slider("High Vol Percentile", 60, 90, 70, 5,
                                  help="Above this percentile = high volatility regime")
        vol_low_rr = st.slider("Low Vol R:R Target", 0.5, 1.5, 1.0, 0.1,
                                help="Reduced R:R target during low volatility (easier to hit)")
        vol_low_max_trades = st.selectbox("Low Vol Max Trades/Day", [1, 2], index=0,
                                           help="Fewer trades during low vol = higher quality only")
    else:
        vol_atr_period = 10
        vol_low_pct = 30
        vol_high_pct = 70
        vol_low_rr = 1.0
        vol_low_max_trades = 1

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
        'partial_tp_pct': partial_tp_pct,
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
        'use_stop_after_loss': use_stop_after_loss,
        'use_opening_range_filter': use_opening_range_filter,
        'use_volatility_regime': use_vol_regime,
        'vol_atr_period': vol_atr_period,
        'vol_low_percentile': vol_low_pct,
        'vol_high_percentile': vol_high_pct,
        'vol_low_rr_target': vol_low_rr,
        'vol_low_max_trades': vol_low_max_trades,
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

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("Total Trades", metrics['total_trades'])
    with col2:
        wr = metrics['win_rate']
        st.metric("Win Rate", f"{wr}%", delta=f"{'Strong' if wr >= 60 else 'Moderate' if wr >= 50 else 'Low'}")
    with col3:
        pnl = metrics['total_pnl_pts']
        st.metric("Total P&L", f"{pnl:+.1f} pts", delta=f"${metrics['total_pnl_dollars']:+.0f}")
    with col4:
        st.metric("Profit Factor", f"{metrics['profit_factor']:.2f}")
    with col5:
        st.metric("Max Drawdown", f"{metrics['max_drawdown_pts']:.1f} pts")
    with col6:
        st.metric("Avg R:R Wins", f"{metrics['avg_rr_on_wins']:.2f}")

    col7, col8, col9, col10, col11, col12 = st.columns(6)
    with col7:
        st.metric("Trades/Week", f"~{metrics.get('trades_per_week', 0):.1f}")
    with col8:
        st.metric("Trades/Month", f"~{metrics.get('trades_per_month', 0):.0f}")
    with col9:
        st.metric("Max Loss Streak", f"{metrics.get('max_consecutive_losses', 0)}")
    with col10:
        wd = metrics['winning_days']
        td = max(metrics['total_trading_days'], 1)
        st.metric("Win Days", f"{wd}/{td}", delta=f"{wd/td*100:.0f}%")
    with col11:
        st.metric("Avg Daily P&L", f"{metrics['avg_daily_pnl']:+.1f} pts")
    with col12:
        st.metric("Best Day", f"{metrics['best_day_pts']:+.1f} pts")

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "Equity Curve", "Trade Log", "Statistics", "Consistance Mensuelle", "Statistical Analysis", "Daily Analysis", "Economic Calendar", "Optimizer", "Strategy Guide"
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
                line=dict(color=COLORS['green'], width=2),
                marker=dict(
                    size=6,
                    color=[COLORS['green'] if p > 0 else COLORS['red'] for p in trades_df['pnl_pts']],
                ),
                hovertemplate="Trade #%{x}<br>Cum P&L: %{y:.1f} pts<extra></extra>"
            ),
            row=1, col=1,
        )

        fig.add_hline(y=0, line_dash="dash", line_color="#30363d", opacity=0.8, row=1, col=1)

        fig.add_trace(
            go.Bar(
                x=list(range(len(trades_df))),
                y=trades_df['drawdown'],
                name='Drawdown',
                marker_color=COLORS['red'],
                opacity=0.5,
            ),
            row=2, col=1,
        )

        fig.update_layout(
            **{k: v for k, v in CHART_LAYOUT.items() if k != 'legend'},
            height=550, showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11, color='#c9d1d9')),
        )
        fig.update_xaxes(title_text="Trade Number", row=2, col=1, gridcolor='#21262d')
        fig.update_yaxes(title_text="Points", row=1, col=1, gridcolor='#21262d')
        fig.update_yaxes(title_text="Points", row=2, col=1, gridcolor='#21262d')

        st.plotly_chart(fig, use_container_width=True)

        fig_time = go.Figure()
        fig_time.add_trace(go.Scatter(
            x=trades_df['entry_time'],
            y=trades_df['cum_pnl_pts'],
            mode='lines+markers',
            name='P&L over time',
            line=dict(color=COLORS['green'], width=2),
            marker=dict(
                size=5,
                color=[COLORS['green'] if r == 'WIN' else COLORS['red'] if r == 'LOSS' else COLORS['orange']
                       for r in trades_df['result']],
            ),
        ))
        fig_time.add_hline(y=0, line_dash="dash", line_color="#30363d", opacity=0.8)
        fig_time.update_layout(
            **CHART_LAYOUT,
            height=320,
            xaxis_title="Date",
            yaxis_title="Cumulative P&L (pts)",
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
            'entry_time', 'direction', 'result', 'entry', 'sl', 'tp', 'tp_partial',
            'exit_price', 'risk_pts', 'pnl_pts', 'pnl_dollars', 'rr_achieved',
            'tp_pct_reached', 'fvg_size', 'h1_bias', 'target_mode'
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
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### Trade Results")
            st.metric("Wins", metrics['wins'])
            st.metric("Losses", metrics['losses'])
            st.metric("Breakevens", metrics['breakevens'])
            st.metric("EOD Exits", metrics['eod_exits'])

        with col2:
            st.markdown("#### P&L Analysis")
            st.metric("Total P&L", f"{metrics['total_pnl_pts']:+.2f} pts")
            st.metric("Avg Win", f"{metrics['avg_win_pts']:+.2f} pts")
            st.metric("Avg Loss", f"{metrics['avg_loss_pts']:.2f} pts")
            st.metric("Profit Factor", f"{metrics['profit_factor']:.2f}")

        with col3:
            st.markdown("#### Streaks & Risk")
            st.metric("Max Drawdown", f"{metrics['max_drawdown_pts']:.2f} pts")
            st.metric("Max DD ($)", f"${metrics['max_drawdown_dollars']:.2f}")
            st.metric("Max Consec. Wins", metrics['max_consecutive_wins'])
            st.metric("Max Consec. Losses", metrics['max_consecutive_losses'])

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        dist_col, pnl_col = st.columns(2)

        with dist_col:
            st.markdown("#### Result Distribution")
            fig_dist = go.Figure()
            results_counts = trades_df['result'].value_counts()
            colors_map = {'WIN': COLORS['green'], 'LOSS': COLORS['red'], 'BE': COLORS['orange'], 'EOD': COLORS['gray']}
            fig_dist.add_trace(go.Pie(
                labels=results_counts.index,
                values=results_counts.values,
                marker_colors=[colors_map.get(r, COLORS['gray']) for r in results_counts.index],
                hole=0.45,
                textinfo='label+percent+value',
                textfont=dict(size=12),
            ))
            fig_dist.update_layout(**CHART_LAYOUT, height=320)
            st.plotly_chart(fig_dist, use_container_width=True)

        with pnl_col:
            st.markdown("#### P&L Distribution")
            fig_pnl = go.Figure()
            fig_pnl.add_trace(go.Histogram(
                x=trades_df['pnl_pts'], nbinsx=25,
                marker_color=COLORS['green'], opacity=0.7,
                marker_line=dict(color=COLORS['green'], width=0.5),
            ))
            fig_pnl.add_vline(x=0, line_dash="dash", line_color="#30363d", opacity=0.8)
            fig_pnl.update_layout(
                **CHART_LAYOUT, height=320,
                xaxis_title="P&L (Points)", yaxis_title="Frequency",
            )
            st.plotly_chart(fig_pnl, use_container_width=True)

        if 'direction' in trades_df.columns:
            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
            st.markdown("#### Performance by Direction")
            dir_stats = trades_df.groupby('direction').agg(
                count=('pnl_pts', 'count'),
                total_pnl=('pnl_pts', 'sum'),
                avg_pnl=('pnl_pts', 'mean'),
                wins=('result', lambda x: (x == 'WIN').sum()),
            ).reset_index()
            dir_stats['losses'] = dir_stats.apply(lambda r: (trades_df[trades_df['direction']==r['direction']]['result']=='LOSS').sum(), axis=1)
            dir_stats['decisive'] = dir_stats['wins'] + dir_stats['losses']
            dir_stats['win_rate'] = (dir_stats['wins'] / dir_stats['decisive'] * 100).where(dir_stats['decisive'] > 0, 0).round(1)
            st.dataframe(dir_stats, use_container_width=True)

    with tab4:
        st.subheader("Consistance Mensuelle")
        st.caption("Analyse de la regularite mensuelle — l'objectif est d'atteindre 60%+ de win rate chaque mois")

        if 'entry_time' in trades_df.columns:
            trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
            if hasattr(trades_df['entry_time'].dt, 'tz') and trades_df['entry_time'].dt.tz is not None:
                _et_naive = trades_df['entry_time'].dt.tz_localize(None)
            else:
                _et_naive = trades_df['entry_time']
            trades_df['year_month'] = _et_naive.dt.to_period('M').astype(str)
        elif 'trade_date' in trades_df.columns:
            trades_df['trade_date'] = pd.to_datetime(trades_df['trade_date'])
            trades_df['year_month'] = trades_df['trade_date'].dt.to_period('M').astype(str)

        if 'year_month' in trades_df.columns:
            monthly_grp = trades_df.groupby('year_month').agg(
                trades=('pnl_pts', 'count'),
                wins=('result', lambda x: (x == 'WIN').sum()),
                losses=('result', lambda x: (x == 'LOSS').sum()),
                bes=('result', lambda x: (x == 'BE').sum()),
                pnl=('pnl_pts', 'sum'),
            ).reset_index()
            monthly_grp['decisive'] = monthly_grp['wins'] + monthly_grp['losses']
            monthly_grp['win_rate'] = (monthly_grp['wins'] / monthly_grp['decisive'] * 100).where(monthly_grp['decisive'] > 0, 0).round(1)

            qualified = monthly_grp[monthly_grp['decisive'] >= 3]
            q_wrs = qualified['win_rate'].values if len(qualified) > 0 else np.array([])
            months_at_target = int((q_wrs >= 60).sum()) if len(q_wrs) > 0 else 0
            months_below_60 = int((q_wrs < 60).sum()) if len(q_wrs) > 0 else 0
            total_qualified = len(q_wrs)
            consistency_score = round(months_at_target / total_qualified * 100, 1) if total_qualified > 0 else 0
            wr_floor = round(float(q_wrs.min()), 1) if len(q_wrs) > 0 else 0
            wr_ceiling = round(float(q_wrs.max()), 1) if len(q_wrs) > 0 else 0

            mc1, mc2, mc3, mc4, mc5 = st.columns(5)
            with mc1:
                st.metric("Score de Consistance", f"{consistency_score}%")
            with mc2:
                st.metric("Mois >= 60% WR", f"{months_at_target}/{total_qualified}")
            with mc3:
                st.metric("Mois < 60% WR", f"{months_below_60}",
                           delta="OK" if months_below_60 == 0 else f"-{months_below_60}",
                           delta_color="normal" if months_below_60 == 0 else "inverse")
            with mc4:
                st.metric("WR Plancher", f"{wr_floor}%")
            with mc5:
                st.metric("WR Plafond", f"{wr_ceiling}%")

            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

            fig_wr = go.Figure()
            bar_colors = []
            for _, row in monthly_grp.iterrows():
                if row['trades'] < 3:
                    bar_colors.append(COLORS['gray'])
                elif row['win_rate'] >= 60:
                    bar_colors.append(COLORS['green'])
                else:
                    bar_colors.append(COLORS['red'])

            fig_wr.add_trace(go.Bar(
                x=monthly_grp['year_month'],
                y=monthly_grp['win_rate'],
                marker_color=bar_colors,
                text=[f"{wr:.0f}%" for wr in monthly_grp['win_rate']],
                textposition='outside',
                textfont=dict(size=9, color='#c9d1d9'),
                name='Win Rate %',
            ))
            fig_wr.add_hline(y=60, line_dash="dash", line_color=COLORS['green'],
                              annotation_text="Objectif 60%", annotation_position="top right",
                              annotation_font_color=COLORS['green'], line_width=2)
            fig_wr.update_layout(
                **CHART_LAYOUT, height=380,
                title="Win Rate Mensuel",
                yaxis_title="Win Rate %",
                yaxis_range=[0, max(100, monthly_grp['win_rate'].max() + 10)],
            )
            st.plotly_chart(fig_wr, use_container_width=True)

            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

            pnl_col_chart, count_col_chart = st.columns(2)

            with pnl_col_chart:
                fig_pnl_m = go.Figure()
                fig_pnl_m.add_trace(go.Bar(
                    x=monthly_grp['year_month'],
                    y=monthly_grp['pnl'],
                    marker_color=[COLORS['green'] if p > 0 else COLORS['red'] for p in monthly_grp['pnl']],
                    name='P&L (pts)',
                ))
                fig_pnl_m.update_layout(
                    **CHART_LAYOUT, height=340,
                    title="P&L Mensuel (pts)",
                    yaxis_title="P&L (pts)",
                )
                st.plotly_chart(fig_pnl_m, use_container_width=True)

            with count_col_chart:
                fig_cnt = go.Figure()
                fig_cnt.add_trace(go.Bar(
                    x=monthly_grp['year_month'],
                    y=monthly_grp['trades'],
                    marker_color=COLORS['blue'],
                    name='Trades',
                ))
                fig_cnt.add_hline(y=3, line_dash="dot", line_color=COLORS['orange'],
                                   annotation_text="Min 3 trades", annotation_position="top right",
                                   annotation_font_color=COLORS['orange'], line_width=1)
                fig_cnt.update_layout(
                    **CHART_LAYOUT, height=340,
                    title="Nombre de Trades par Mois",
                    yaxis_title="Trades",
                )
                st.plotly_chart(fig_cnt, use_container_width=True)

            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
            st.markdown("#### Detail Mensuel")

            display_monthly = monthly_grp.rename(columns={
                'year_month': 'Mois',
                'trades': 'Trades',
                'wins': 'Gains',
                'win_rate': 'WR %',
                'pnl': 'P&L (pts)',
            })
            st.dataframe(display_monthly, use_container_width=True, hide_index=True)
        else:
            st.info("Donnees insuffisantes pour l'analyse mensuelle.")

    with tab5:
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
                       name='P&L (pts)', marker_color=[COLORS['green'] if p > 0 else COLORS['red'] for p in yearly_df['pnl']]),
                secondary_y=False,
            )
            fig_yearly.add_trace(
                go.Scatter(x=yearly_df['year'].astype(str), y=yearly_df['win_rate'],
                           name='Win Rate %', mode='lines+markers',
                           line=dict(color=COLORS['line'], width=2), marker=dict(size=8)),
                secondary_y=True,
            )
            fig_yearly.update_layout(**CHART_LAYOUT, height=280, title="Yearly Performance")
            fig_yearly.update_yaxes(title_text="P&L (pts)", secondary_y=False, gridcolor='#21262d')
            fig_yearly.update_yaxes(title_text="Win Rate %", secondary_y=True, gridcolor='#21262d')
            st.plotly_chart(fig_yearly, use_container_width=True)

        monthly_df = pd.DataFrame(cohort['monthly'])
        if not monthly_df.empty:
            fig_monthly = go.Figure()
            fig_monthly.add_trace(go.Bar(
                x=monthly_df['year_month'], y=monthly_df['pnl'],
                name='Monthly P&L',
                marker_color=[COLORS['green'] if p > 0 else COLORS['red'] for p in monthly_df['pnl']],
            ))
            fig_monthly.add_trace(go.Scatter(
                x=monthly_df['year_month'], y=monthly_df['win_rate'],
                name='Win Rate %', yaxis='y2', mode='lines+markers',
                line=dict(color=COLORS['line'], width=2), marker=dict(size=4),
            ))
            fig_monthly.update_layout(
                **CHART_BASE, height=320, title="Monthly Breakdown",
                xaxis=dict(**AXIS_DEFAULTS),
                yaxis=dict(title='P&L (pts)', **AXIS_DEFAULTS),
                yaxis2=dict(title='Win Rate %', overlaying='y', side='right', **AXIS_DEFAULTS),
            )
            st.plotly_chart(fig_monthly, use_container_width=True)

        dow_df = pd.DataFrame(cohort['by_day_of_week'])
        if not dow_df.empty:
            fig_dow = go.Figure()
            fig_dow.add_trace(go.Bar(
                x=dow_df['day_of_week'], y=dow_df['pnl'],
                name='P&L', marker_color=[COLORS['green'] if p > 0 else COLORS['red'] for p in dow_df['pnl']],
            ))
            fig_dow.add_trace(go.Scatter(
                x=dow_df['day_of_week'], y=dow_df['win_rate'],
                name='Win Rate %', yaxis='y2', mode='lines+markers',
                line=dict(color=COLORS['line'], width=2), marker=dict(size=8),
            ))
            fig_dow.update_layout(
                **CHART_BASE, height=280, title="Performance by Day of Week",
                xaxis=dict(**AXIS_DEFAULTS),
                yaxis=dict(title='P&L (pts)', **AXIS_DEFAULTS),
                yaxis2=dict(title='Win Rate %', overlaying='y', side='right', **AXIS_DEFAULTS),
            )
            st.plotly_chart(fig_dow, use_container_width=True)

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown("### Monthly Consistency Analysis")
        st.caption("Target: 60%+ win rate every month — stability is more important than peak performance")

        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        with mc1:
            cons_pct = cohort.get('consistency_pct', 0)
            st.metric("Months at 60%+ WR",
                       f"{cohort.get('months_at_target', 0)}/{cohort.get('total_qualified_months', 0)}",
                       delta=f"{cons_pct}%")
        with mc2:
            m60 = cohort.get('months_below_60', 0)
            st.metric("Months Below 60%", f"{m60}",
                       delta="OK" if m60 == 0 else f"-{m60} months",
                       delta_color="normal" if m60 == 0 else "inverse")
        with mc3:
            st.metric("WR Floor", f"{cohort.get('monthly_wr_floor', 0)}%",
                       help="Lowest monthly win rate (months with 3+ trades)")
        with mc4:
            st.metric("WR Ceiling", f"{cohort.get('monthly_wr_ceiling', 0)}%",
                       help="Highest monthly win rate")
        with mc5:
            neg = cohort.get('negative_pnl_months', 0)
            st.metric("Negative P&L Months", f"{neg}",
                       delta="None" if neg == 0 else f"-{neg}",
                       delta_color="normal" if neg == 0 else "inverse")

        monthly_df_stat = pd.DataFrame(cohort['monthly'])
        if not monthly_df_stat.empty and len(monthly_df_stat) > 1:
            fig_cons = go.Figure()

            colors_wr = []
            for wr in monthly_df_stat['win_rate']:
                if wr >= 60:
                    colors_wr.append(COLORS['green'])
                elif wr >= 50:
                    colors_wr.append(COLORS['orange'])
                else:
                    colors_wr.append(COLORS['red'])

            fig_cons.add_trace(go.Bar(
                x=monthly_df_stat['year_month'], y=monthly_df_stat['win_rate'],
                name='Monthly WR%', marker_color=colors_wr,
                text=[f"{wr:.0f}%" for wr in monthly_df_stat['win_rate']],
                textposition='outside', textfont=dict(size=9, color='#c9d1d9'),
            ))

            fig_cons.add_hline(y=60, line_dash="dash", line_color=COLORS['green'],
                                annotation_text="60% Target", annotation_position="top right",
                                annotation_font_color=COLORS['green'], line_width=2)

            fig_cons.add_hline(y=50, line_dash="dot", line_color=COLORS['orange'],
                                annotation_text="50% Breakeven", annotation_position="bottom right",
                                annotation_font_color=COLORS['orange'], line_width=1)

            fig_cons.update_layout(
                **CHART_LAYOUT, height=340,
                title="Monthly Win Rate — Consistency Tracker",
                yaxis_title="Win Rate %",
                yaxis_range=[0, max(100, monthly_df_stat['win_rate'].max() + 10)],
            )
            st.plotly_chart(fig_cons, use_container_width=True)

            below_60_months = monthly_df_stat[monthly_df_stat['win_rate'] < 60]
            if len(below_60_months) > 0 and len(below_60_months) <= 10:
                st.markdown("**Months below 60% WR:**")
                for _, row in below_60_months.iterrows():
                    st.markdown(f"- `{row['year_month']}`: **{row['win_rate']:.1f}%** ({row['trades']} trades, {row['pnl']:+.1f} pts)")

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
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

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
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
                        marker_color=COLORS['green'],
                    ))
                    fig_ws.update_layout(**CHART_LAYOUT, height=240,
                                          title="Win Streak Distribution",
                                          xaxis_title="Streak Length", yaxis_title="Count")
                    st.plotly_chart(fig_ws, use_container_width=True)
            with sd2:
                if loss_dist:
                    fig_ls = go.Figure(go.Bar(
                        x=[str(k) for k in loss_dist.keys()],
                        y=list(loss_dist.values()),
                        marker_color=COLORS['red'],
                    ))
                    fig_ls.update_layout(**CHART_LAYOUT, height=240,
                                          title="Loss Streak Distribution",
                                          xaxis_title="Streak Length", yaxis_title="Count")
                    st.plotly_chart(fig_ls, use_container_width=True)

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
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
            title={'text': "Win Rate vs Breakeven", 'font': {'size': 14, 'color': '#c9d1d9'}},
            number={'font': {'size': 28, 'color': '#f0f6fc'}},
            delta={'reference': rr_analysis['breakeven_win_rate'], 'suffix': '%'},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': '#8b949e'},
                'bar': {'color': COLORS['green']},
                'bgcolor': '#161b22',
                'bordercolor': '#21262d',
                'steps': [
                    {'range': [0, rr_analysis['breakeven_win_rate']], 'color': 'rgba(255,71,87,0.2)'},
                    {'range': [rr_analysis['breakeven_win_rate'], 100], 'color': 'rgba(0,212,170,0.1)'},
                ],
                'threshold': {
                    'line': {'color': COLORS['line'], 'width': 3},
                    'thickness': 0.75,
                    'value': rr_analysis['breakeven_win_rate'],
                },
            },
        ))
        fig_rr.update_layout(**CHART_LAYOUT, height=280)
        st.plotly_chart(fig_rr, use_container_width=True)

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
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
                marker_color=[COLORS['green'] if p > 0 else COLORS['red'] for p in mr_df['pnl']],
            ))
            cum_monthly = mr_df['pnl'].cumsum()
            fig_mr.add_trace(go.Scatter(
                x=mr_df['month'], y=cum_monthly,
                mode='lines', name='Cumulative',
                line=dict(color=COLORS['line'], width=2),
            ))
            fig_mr.update_layout(
                **CHART_LAYOUT, height=320,
                title="Monthly P&L Distribution",
                xaxis_title="Month", yaxis_title="P&L (pts)",
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

    with tab6:
        daily = trades_df.groupby('trade_date').agg(
            trades=('pnl_pts', 'count'),
            pnl=('pnl_pts', 'sum'),
            wins=('result', lambda x: (x == 'WIN').sum()),
            losses=('result', lambda x: (x == 'LOSS').sum()),
        ).reset_index()
        daily['decisive'] = daily['wins'] + daily['losses']
        daily['win_rate'] = (daily['wins'] / daily['decisive'] * 100).where(daily['decisive'] > 0, 0).round(1)
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
                marker_color=[COLORS['green'] if p > 0 else COLORS['red'] for p in daily['pnl']],
            ),
            row=1, col=1,
        )

        fig_daily.add_trace(
            go.Scatter(
                x=daily['trade_date'].astype(str),
                y=daily['cum_pnl'],
                mode='lines+markers',
                name='Cumulative',
                line=dict(color=COLORS['line'], width=2),
                marker=dict(size=4),
            ),
            row=2, col=1,
        )

        fig_daily.update_layout(
            **CHART_LAYOUT, height=520, showlegend=True,
        )
        fig_daily.update_xaxes(gridcolor='#21262d')
        fig_daily.update_yaxes(gridcolor='#21262d')
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

    with tab7:
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

            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
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

            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
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

            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
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

    with tab8:
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
                st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
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
                st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
                st.markdown("#### Parameter Sensitivity Analysis")

                for param in available_param_cols:
                    if top_saved[param].nunique() > 1:
                        fig_sens = go.Figure()
                        fig_sens.add_trace(go.Box(
                            x=top_saved[param].head(20).astype(str),
                            y=top_saved['total_pnl_pts'].head(20),
                            name=param,
                            marker_color=COLORS['green'],
                        ))
                        fig_sens.update_layout(
                            **CHART_LAYOUT, height=240,
                            title=f"P&L by {param} (Top 20 configs)",
                            xaxis_title=param,
                            yaxis_title="Total P&L (pts)",
                        )
                        st.plotly_chart(fig_sens, use_container_width=True)
        else:
            st.info("No optimization results found yet.")

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
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

    with tab9:
        st.subheader("Multi-TF IFVG Strategy — Guide Complet de Prise de Position")
        st.caption("Toutes les phases, analyses et conditions requises, de la preparation pre-session jusqu'a la gestion active du trade")

        config_used = st.session_state.get('config', {})

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
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

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
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

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
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

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
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

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
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

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
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

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
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

**Contrainte temporelle** : L'inversion doit se produire dans les **8 barres M15** (2h00) suivant la creation du FVG. Au-dela, le FVG est considere comme expire.

**Alignement obligatoire** : L'inversion ne produit un signal que si elle est **dans la direction du biais H1**.
""")

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
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

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown("## Phase 8 : Recherche d'Entree sur M1 (Killzone)")
        st.markdown("""
**Objectif** : Trouver le point d'entree precis sur le timeframe 1 minute, en attendant un retracement dans la zone IFVG.

**Fenetre d'entree** :
- **Debut** : 10:05 ET (par defaut, retarde de 35 min apres l'ouverture pour filtrer la volatilite initiale)
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

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
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

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
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

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
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

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown("## Phase 11 : Gestion Active du Trade")
        st.markdown("""
**Objectif** : Proteger le capital et securiser les profits en cours de trade.

**11a. Prise de Profit Partielle (TP Partiel)** :
- **Principe** : Le trade est ferme en WIN des que le prix atteint **60%** du TP total
  - BUY : high atteint `entry + (distance_TP x 60%)`
  - SELL : low atteint `entry - (distance_TP x 60%)`
- **Avantage** : Garantit un profit significatif a chaque WIN (minimum 60% du TP cible)
- **Correlation WR/PnL** : Avec un WR de 60% et un gain garanti de 60% du TP, le PnL est mathematiquement positif
- **Calcul** : Avec RR 1.2 → gain par WIN = 0.6 x 1.2R = 0.72R, perte = 1R. EV a 60% WR = +0.03R/trade
- **Configurable** : Le % est ajustable via le slider "TP Partiel" (30-100%)

**11b. Breakeven (Protection du capital)** :
- **Declenchement** : Quand le prix bouge de **0.5R** en faveur du trade
  - BUY : high atteint `entry + (risk x 0.5)`
  - SELL : low atteint `entry - (risk x 0.5)`
- **Action** : Le stop loss est deplace a `entry +/- 1 point` (breakeven + 1 pt pour couvrir les frais)
- **Important** : Le breakeven ne s'active que si le **trailing stop n'est pas deja actif**
- Desormais, le trade ne peut plus etre perdant (sauf gap)

**11c. Trailing Stop (Securisation supplementaire)** :
- **Declenchement** : Quand le prix bouge de **0.3R** en faveur
- **Note** : Avec le TP partiel a 60%, le trailing stop sert de filet de securite supplementaire. La plupart des trades se ferment au TP partiel avant que le trailing ne soit necessaire.
- **Fonctionnement** :
  - Le SL suit le meilleur prix atteint, a une distance de **30% du risque initial**
  - BUY : `trailing_SL = best_high - (risk x 30%)`
  - SELL : `trailing_SL = best_low + (risk x 30%)`

**11d. Sortie de fin de journee (EOD)** :
- Si ni le TP ni le SL n'est touche avant la fin de la session
- Le trade est ferme au **dernier close disponible**
- Resultat marque comme "EOD" (End Of Day)

**Priorite de sortie** (evaluee barre par barre) :
1. Stop Loss touche → sortie immediate
2. Take Profit complet (100%) touche → WIN
3. TP Partiel (60%) touche → WIN
4. Mise a jour trailing stop → ajustement du SL
5. Activation breakeven → ajustement du SL
6. Fin de journee → sortie forcee
""")

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown("## Phase 12 : Limites de Position et Regles de Gestion")
        st.markdown("""
**Objectif** : Controler l'exposition et eviter le sur-trading.

| Regle | Valeur par defaut |
|-------|------------------|
| Max trades par jour | **2** |
| Cooldown entre trades | **10 minutes** |
| Chaque IFVG utilise une seule fois | Oui |
| Killzone | **09:30 - 12:00 ET** |
| Entrees a partir de | **10:00 ET** |
| Stop apres perte | **Oui** |

**Logique du cooldown** : Apres chaque trade (gagnant ou perdant), attendre au minimum 10 minutes avant de chercher une nouvelle entree. Empeche les entrees emotionnelles consecutives.

**Utilisation unique des IFVG** : Une fois qu'un IFVG a genere un trade (gagnant ou perdant), il ne peut plus etre utilise. Cela evite de re-entrer sur une zone deja exploitee.
""")

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown("## Phase 12b : Stop Apres Perte")
        st.markdown("""
**Objectif** : Proteger le capital psychologique et financier en arretant de trader apres la premiere perte de la journee.

**Fonctionnement** :
- Si le premier trade de la journee est un **LOSS**, aucune nouvelle entree n'est recherchee pour le reste de la session
- Cette regle empeche le **revenge trading** — la tendance a vouloir recuperer une perte en prenant des trades impulsifs
- Active par defaut (`use_stop_after_loss = True`)

**Justification** :
- Les donnees montrent que le deuxieme trade apres une perte a souvent un win rate inferieur
- En limitant a 1 perte max/jour, on reduit significativement le drawdown maximum
- Un trader discipliné accepte la perte et revient le lendemain avec un esprit clair
""")

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown("## Phase 12c : Filtre Opening Range")
        st.markdown("""
**Objectif** : Utiliser la direction de l'Opening Range (09:30 - 09:45 ET) comme filtre de confirmation du biais H1.

**Fonctionnement** :
1. Analyser les 15 premieres minutes de la session reguliere (09:30 - 09:45 ET)
2. Calculer : `or_body = close_09:45 - open_09:30`
3. Calculer : `or_range = high_09:45 - low_09:30`
4. Si le body represente plus de **30%** du range → l'Opening Range a une direction claire
5. Direction OR : **BUY** si body > 0, **SELL** si body < 0

**Condition** :
- La direction de l'Opening Range doit etre **identique** au biais H1
- Si l'OR va dans le sens contraire du biais H1 → **pas de trade ce jour-la**
- Desactive par defaut (`use_opening_range_filter = False`) — trop restrictif pour la frequence cible de ~3 trades/semaine

**Logique** :
- L'Opening Range capture l'intention initiale des participants institutionnels
- Si les 15 premieres minutes vont dans le meme sens que le biais H1, c'est une **double confirmation**
- Un conflit entre l'OR et le biais H1 signale une journee potentiellement indecise
""")

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
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

ENTREE (M1 — a partir de 10:00 ET)
================================
[12] Verifier filtre Opening Range si active (direction OR = biais H1)
[13] Scanner chaque bougie M1 dans la killzone (09:30-12:00 ET)
[14] Verifier retracement dans la zone IFVG (penetration >= 60%)
[15] Valider la bougie de confirmation M1 (corps significatif + rejet)
[16] Calculer entry/SL/TP et verifier que le risque est entre 5-25 pts

GESTION DU TRADE
================================
[17] Surveiller barre par barre : SL → TP → trailing → breakeven → EOD
[18] Respecter max 2 trades/jour et cooldown 10 min
[19] Stop apres perte : arreter si premier trade est un LOSS
```
""")

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown("## Parametres Optimises (Configuration Actuelle)")
        if config_used:
            param_display = {
                'Taille min FVG': f"{config_used.get('min_fvg_size', 3.0)} pts",
                'Age max FVG (barres M15)': f"{config_used.get('max_fvg_age_m15', 12)}",
                'R:R cible': f"{config_used.get('rr_target', 1.2)}",
                'Risque max': f"{config_used.get('max_risk_pts', 25.0)} pts",
                'Risque min': f"{config_used.get('min_risk_pts', 5.0)} pts",
                'Max trades/jour': f"{config_used.get('max_trades_per_day', 2)}",
                'Killzone': f"{config_used.get('killzone_start', '09:30')} - {config_used.get('killzone_end', '12:00')} ET",
                'Debut des entrees': f"{config_used.get('entry_start_time', '10:00')} ET",
                'Cooldown': f"{config_used.get('cooldown_minutes', 10)} min",
                'Retracement': f"{config_used.get('retracement_pct', 60)}%",
                'Breakeven trigger': f"{config_used.get('be_trigger_rr', 0.5)}R",
                'Definition WIN': 'TP touche uniquement',
                'Trailing trigger': f"{config_used.get('trail_trigger_rr', 0.3)}R",
                'Trailing offset': f"{config_used.get('trail_offset_pct', 30)}%",
                'Displacement body min': f"{config_used.get('min_displacement_body_pct', 55)}%",
                'Displacement taille min': f"{config_used.get('min_displacement_size', 3.5)} pts",
                'Mode target': config_used.get('target_mode', 'fixed_rr'),
                'Stop apres perte': 'Oui' if config_used.get('use_stop_after_loss', True) else 'Non',
                'Filtre Opening Range': 'Oui' if config_used.get('use_opening_range_filter', False) else 'Non',
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
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    lc, rc = st.columns([2, 1])
    with lc:
        st.markdown("""
#### How It Works

This backtester implements a **Multi-Timeframe Inversed Fair Value Gap (IFVG)** strategy on MNQ futures, designed to capture high-probability setups during the US market killzone.

**The process in 5 steps:**

1. **H1 Bias** — Determine directional bias from the hourly chart before 9:30 ET
2. **M15 FVG Detection** — Identify institutional price imbalances on the 15-minute timeframe
3. **Inversion Signal** — Wait for price to trade through and close beyond the FVG
4. **M1 Precision Entry** — Enter on a confirmed retracement into the inverted zone
5. **Managed Exit** — Breakeven protection, trailing stop, and fixed R:R or liquidity targets

Configure parameters in the sidebar, then click **Run Backtest** to analyze performance.
""")
    with rc:
        st.markdown("#### Data Available")
        if data_files:
            for lbl in file_labels:
                st.markdown(f"- `{lbl}`")
            st.caption(f"{len(data_files)} month(s) loaded")
        else:
            st.warning("No data files found.")
