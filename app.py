import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

from data_loader import list_data_files, load_data, get_active_contract
from ifvg_strategy import IFVGStrategy

st.set_page_config(
    page_title="MNQ IFVG Backtester",
    page_icon="📊",
    layout="wide",
)

st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e1e2e 0%, #2d2d44 100%);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #3d3d5c;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
    }
    .metric-label {
        font-size: 13px;
        color: #888;
        text-transform: uppercase;
    }
    .positive { color: #00d4aa; }
    .negative { color: #ff4757; }
    .neutral { color: #ffa502; }
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #2d3561;
        border-radius: 10px;
        padding: 15px;
    }
</style>
""", unsafe_allow_html=True)

st.title("MNQ Futures - IFVG Strategy Backtester")
st.caption("Inversed Fair Value Gap strategy on Micro E-mini Nasdaq | US Killzone 9:30-11:00 ET")

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
                         help="Minimum gap size to qualify as a Fair Value Gap")
    max_fvg_age = st.slider("Max FVG Age (minutes)", 5, 120, 30, 5,
                             help="Maximum age of FVG before it expires")
    rr_target = st.slider("Risk:Reward Target", 1.0, 5.0, 2.5, 0.5,
                           help="Target R:R ratio for take profit")
    max_risk = st.slider("Max Risk (pts)", 10.0, 100.0, 50.0, 5.0,
                          help="Maximum allowed risk per trade in points")
    min_risk = st.slider("Min Risk (pts)", 1.0, 20.0, 5.0, 1.0,
                          help="Minimum risk to avoid noise trades")
    max_trades = st.slider("Max Trades / Day", 1, 6, 3, 1)
    displacement_min = st.slider("Min Displacement (pts)", 0.0, 15.0, 3.0, 0.5,
                                  help="Minimum candle body size to confirm IFVG")
    cooldown = st.slider("Cooldown (minutes)", 0, 30, 5, 1,
                          help="Minutes between trades")

    st.subheader("Killzone")
    kz_start = st.text_input("Start (ET)", "09:30")
    kz_end = st.text_input("End (ET)", "11:00")

    st.subheader("Risk Management")
    use_be = st.checkbox("Breakeven Protection", value=True,
                         help="Move SL to breakeven after trade reaches trigger R")
    be_trigger = st.slider("BE Trigger (xR)", 0.5, 3.0, 1.5, 0.1) if use_be else 1.5
    contract_value = st.number_input("Point Value ($)", value=2.0, step=0.5,
                                      help="Dollar value per point (MNQ = $2)")

    run_button = st.button("Run Backtest", type="primary", use_container_width=True)

if run_button:
    config = {
        'min_fvg_size': min_fvg,
        'max_fvg_age': max_fvg_age,
        'rr_target': rr_target,
        'max_risk_pts': max_risk,
        'min_risk_pts': min_risk,
        'displacement_min': displacement_min,
        'cooldown_minutes': cooldown,
        'max_trades_per_day': max_trades,
        'killzone_start': kz_start,
        'killzone_end': kz_end,
        'use_be': use_be,
        'be_trigger_rr': be_trigger,
        'contract_value': contract_value,
    }

    with st.spinner("Loading data..."):
        raw_df = load_data(selected_files)
        df = get_active_contract(raw_df)
        st.session_state['data_loaded'] = True
        st.session_state['df_shape'] = df.shape

    with st.spinner("Running IFVG backtest..."):
        strategy = IFVGStrategy(config)
        results = strategy.run_backtest(df)
        st.session_state['results'] = results
        st.session_state['config'] = config

if 'results' in st.session_state:
    results = st.session_state['results']
    trades_df = results['trades']
    metrics = results['metrics']

    if trades_df.empty:
        st.warning("No trades were generated. Try adjusting the strategy parameters (lower Min FVG Size, increase Max FVG Age, or widen the killzone).")
        st.stop()

    st.header("Performance Overview")

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("Total Trades", metrics['total_trades'])
    with col2:
        st.metric("Win Rate", f"{metrics['win_rate']}%")
    with col3:
        pnl_color = "normal" if metrics['total_pnl_pts'] >= 0 else "inverse"
        st.metric("Total P&L (pts)", f"{metrics['total_pnl_pts']:+.1f}")
    with col4:
        st.metric("Total P&L ($)", f"${metrics['total_pnl_dollars']:+.2f}")
    with col5:
        st.metric("Profit Factor", f"{metrics['profit_factor']:.2f}")
    with col6:
        st.metric("Max Drawdown", f"{metrics['max_drawdown_pts']:.1f} pts")

    tab1, tab2, tab3, tab4 = st.tabs(["Equity Curve", "Trade Log", "Statistics", "Daily Analysis"])

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
                    size=6,
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
                hovertemplate="Trade #%{x}<br>DD: %{y:.1f} pts<extra></extra>"
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
                size=5,
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
            'fvg_type', 'fvg_size', 'fvg_age_min'
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
        colors = {'WIN': '#00d4aa', 'LOSS': '#ff4757', 'BE': '#ffa502', 'EOD': '#747d8c'}
        fig_dist.add_trace(go.Pie(
            labels=results_counts.index,
            values=results_counts.values,
            marker_colors=[colors.get(r, '#747d8c') for r in results_counts.index],
            hole=0.4,
            textinfo='label+percent+value',
        ))
        fig_dist.update_layout(
            height=350,
            template='plotly_dark',
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig_dist, use_container_width=True)

        st.subheader("P&L Distribution")
        fig_pnl = go.Figure()
        fig_pnl.add_trace(go.Histogram(
            x=trades_df['pnl_pts'],
            nbinsx=30,
            marker_color='#00d4aa',
            opacity=0.7,
        ))
        fig_pnl.add_vline(x=0, line_dash="dash", line_color="white", opacity=0.5)
        fig_pnl.update_layout(
            height=300,
            template='plotly_dark',
            xaxis_title="P&L (Points)",
            yaxis_title="Frequency",
            margin=dict(l=50, r=20, t=20, b=30),
        )
        st.plotly_chart(fig_pnl, use_container_width=True)

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
                hovertemplate="Date: %{x}<br>P&L: %{y:.1f} pts<extra></extra>",
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
            height=550,
            template='plotly_dark',
            showlegend=True,
            margin=dict(l=50, r=20, t=40, b=30),
        )
        st.plotly_chart(fig_daily, use_container_width=True)

        st.markdown("#### Day-by-Day Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Trading Days", metrics['total_trading_days'])
        with col2:
            st.metric("Winning Days", f"{metrics['winning_days']} ({metrics['winning_days']/max(metrics['total_trading_days'],1)*100:.0f}%)")
        with col3:
            st.metric("Avg Daily P&L", f"{metrics['avg_daily_pnl']:+.2f} pts")

        st.dataframe(daily, use_container_width=True)

        st.subheader("Performance by Direction")
        dir_stats = trades_df.groupby('direction').agg(
            count=('pnl_pts', 'count'),
            total_pnl=('pnl_pts', 'sum'),
            avg_pnl=('pnl_pts', 'mean'),
            wins=('result', lambda x: (x == 'WIN').sum()),
        ).reset_index()
        dir_stats['win_rate'] = (dir_stats['wins'] / dir_stats['count'] * 100).round(1)
        st.dataframe(dir_stats, use_container_width=True)

else:
    st.info("Configure strategy parameters in the sidebar and click **Run Backtest** to start.")

    with st.expander("About the IFVG Strategy"):
        st.markdown("""
        **Inversed Fair Value Gap (IFVG)** is an ICT concept where:

        1. **Fair Value Gap (FVG)** - A 3-candle pattern where a price gap exists between candle 1 and candle 3
           - **Bullish FVG**: Gap between candle 1 high and candle 3 low (price gapped up)
           - **Bearish FVG**: Gap between candle 1 low and candle 3 high (price gapped down)

        2. **Inversion** - When price returns to a previous FVG and trades through it:
           - A bullish FVG that gets broken to the downside becomes a **SELL signal**
           - A bearish FVG that gets broken to the upside becomes a **BUY signal**

        3. **Killzone** - Trades are only taken during the US session open (default 9:30-11:00 ET)

        4. **Risk Management** - Stop loss at FVG extreme, take profit at configured R:R ratio
        """)

    if data_files:
        st.markdown(f"**{len(data_files)} data file(s) available:** {', '.join(file_labels)}")
