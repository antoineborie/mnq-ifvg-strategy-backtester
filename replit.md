# replit.md

## Overview

This is an **MNQ Futures IFVG Strategy Backtester** — a financial backtesting application for testing an Inversed Fair Value Gap (IFVG) trading strategy on Micro E-mini Nasdaq (MNQ) futures contracts. The application provides a Streamlit-based web UI for configuring strategy parameters, running backtests against historical tick/minute data, and visualizing results with Plotly charts.

The strategy focuses on the US market "killzone" session (9:30 AM–12:00 PM Eastern Time) and identifies fair value gaps (FVGs) that get inversed as trade entry signals, with configurable risk/reward management including break-even triggers.

## User Preferences

Preferred communication style: Simple, everyday language.
UI language: French for all user-facing labels, tabs, and documentation within the app.

## System Architecture

### Application Layer
- **Framework**: Streamlit serves as the web application framework, providing the interactive UI for parameter configuration and results display.
- **Entry point**: `app.py` is the main Streamlit application. `main.py` exists but is a basic placeholder and not the primary entry.
- **Rationale**: Streamlit was chosen for rapid prototyping of data-heavy financial dashboards without needing a separate frontend/backend architecture.

### Core Components

1. **`app.py`** — Streamlit UI layer. Handles page config, custom CSS styling, file selection, and rendering of backtest results. Uses Plotly for interactive charting. 9 tabs: Equity Curve, Trade Log, Statistics, Consistance Mensuelle, Statistical Analysis, Daily Analysis, Economic Calendar, Optimizer, Strategy Guide.

2. **`data_loader.py`** — Data ingestion module. Loads historical MNQ futures data from `.pkl` (pickle) files stored in a `data/` directory. Includes compatibility shims (`PlaceholderStringArray`, `CompatStringDtype`, `_fake_pyx_unpickle`) to handle deserialization of pickle files created with different pandas/Cython versions. Also provides `get_active_contract` for futures contract roll management and `list_data_files` for discovering available data files.

3. **`ifvg_strategy.py`** — The core trading strategy engine (`IFVGStrategy` class). Implements:
   - Fair Value Gap detection with configurable minimum size and max age (default 8 bars)
   - Killzone-based session filtering (default 9:30–11:00 ET)
   - Risk/reward targeting with configurable ratios
   - Break-even management triggered at a specified RR level
   - Per-day trade limits (default 1) and cooldown periods
   - Displacement filtering for entry quality
   - Stop After Loss: stops trading after first loss of the day
   - Opening Range Filter: requires opening range bias alignment with H1 bias
   - Day-by-day processing with timezone conversion (UTC → America/New_York)

4. **`optimizer.py`** — Parameter optimization engine with pre-computed day data for speed. Includes consistency-weighted scoring (months_below_60, monthly_wr_std, consistency_score). Fixed params include stop_after_loss and opening_range_filter.

5. **`stat_analysis.py`** — Statistical analysis module with monthly consistency metrics, Monte Carlo simulation, and regime analysis.

6. **`econ_calendar.py`** — Economic calendar module with hardcoded event dates (Jan-May 2025) and impact analysis.

7. **`attached_assets/`** — Reference/legacy files showing earlier iterations of the data loading and market context analysis logic.

### Data Architecture
- **Storage format**: Pandas DataFrames serialized as `.pkl` (pickle) files, organized by contract and month (e.g., `MNQ_2025_12.pkl`).
- **Data location**: Expected in a `data/` directory at the project root.
- **Index**: DateTime index in UTC, converted to Eastern Time for strategy processing.
- **Columns**: Expected to include at minimum `open`, `high`, `low`, `close`, `volume`, and `symbol` (for contract identification).
- **No database**: The project uses flat file storage only. There is no SQL database or ORM involved.

### Strategy Configuration (Optimized Defaults — Feb 2026)
The strategy uses a dictionary-based configuration pattern with optimized defaults targeting **higher R:R with pure TP/SL** on 2024-2026 data:
- `min_fvg_size`: 3.0 points minimum gap size
- `max_fvg_age_m15`: **15 bars** maximum gap age
- `rr_target`: **2.0** risk-to-reward ratio (bigger targets, lets trade breathe)
- `max_risk_pts` / `min_risk_pts`: 25.0 / 5.0 points risk bounds
- `max_trades_per_day`: **2** (more opportunities)
- `contract_value`: $2.00 per point (MNQ specification)
- Break-even: **OFF** (pure TP/SL, no early exit)
- Trailing stop: **OFF** (pure fixed targets, maximizes WR at high RR)
- Displacement filter: **OFF** (too restrictive, reduces trade count without improving WR)
- M1 confirmation candle required
- `killzone_end`: **12:00** ET
- `entry_start_time`: **09:45** ET (earlier entry for more opportunities)
- `use_stop_after_loss`: **False** (allows multiple trades per day)
- `use_opening_range_filter`: **False**
- 5-minute cooldown between trades
- `retracement_pct`: **50%** (balanced entry depth)
- **Data scope**: 2024-2026 only (default file selection)

### Win Rate Calculation (TP-Hit WR — Feb 2026)
- **WIN = only when Take Profit is actually hit** — no partial TP, no trailing stop wins
- **WR = Wins / (Wins + Losses)** — BE and EOD trades excluded from WR denominator
- Pure TP/SL mode: no BE or trailing stop, all trades are WIN or LOSS
- SL hit = LOSS, end-of-day close = EOD
- This strict definition ensures WR reflects real TP accuracy

### Performance Results (Optimized Config — Feb 2026, 2024-2026 data)
- **~397 trades** over 25 months (Jan 2024 – Jan 2026), all decisive (pure TP/SL)
- **~40% WR** (TP-hit only), +922 pts P&L, PF 1.33
- **64% months profitable** (16/25 months with positive PnL)
- Average ~37 pts/month, ~3.7 trades per week
- Key insight: Pure TP/SL (no BE/trailing) maximizes WR at high RR. BE and trailing convert potential TP hits into early exits, reducing WR.
- Trade-off: Higher RR (2.0 vs 0.8) = lower WR (40% vs 62%) but bigger wins per trade

### Project Status
The project is fully implemented with a multi-timeframe IFVG backtest engine and interactive Streamlit dashboard. All core features are complete:
- Multi-TF approach: H1 bias → M15 FVG detection → M15 inversion → M1 retracement entry
- Quality filters: M1 confirmation candles, displacement analysis, trailing stops, delayed entry timing, stop-after-loss, opening range filter
- Market structure analysis: daily/weekly/monthly highs/lows, swing points as liquidity references
- Economic calendar module (econ_calendar.py) with hardcoded event dates (Jan-May 2025) and impact analysis
- Target mode selection: Fixed R:R vs SSL/BSL (liquidity level targeting)
- Dashboard with 9 tabs: Equity Curve, Trade Log, Statistics, Consistance Mensuelle, Statistical Analysis, Daily Analysis, Economic Calendar, Optimizer, Strategy Guide
- Optimizer uses pre-computed confirmation masks for speed (~0.6s/combo) with consistency-weighted scoring
- Strategy Guide fully documented in French

## External Dependencies

### Python Packages
- **Streamlit** — Web application framework and UI
- **Pandas** — Data manipulation and time series handling
- **NumPy** — Numerical computations
- **Plotly** — Interactive charting (via `plotly.graph_objects` and `plotly.subplots`)

### Data Requirements
- Historical MNQ futures data as pickle files in `data/` directory
- Files follow naming convention: `MNQ_{year}_{month}.pkl`
- Data must include OHLCV columns with a UTC DateTimeIndex

### No External Services
- No external APIs, databases, or authentication systems are used
- All data is loaded from local pickle files
- The application runs entirely locally via Streamlit
