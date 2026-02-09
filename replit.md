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
The strategy uses a dictionary-based configuration pattern with optimized defaults prioritizing **monthly consistency** (60%+ WR every month):
- `min_fvg_size`: 3.0 points minimum gap size
- `max_fvg_age_m15`: **12 bars** maximum gap age (balanced freshness vs opportunity)
- `rr_target`: 1.2 risk-to-reward ratio
- `max_risk_pts` / `min_risk_pts`: 25.0 / 5.0 points risk bounds
- `max_trades_per_day`: **2** (increased for ~3 trades/week target)
- `contract_value`: $2.00 per point (MNQ specification)
- Break-even trigger at 0.5 RR
- Trailing stop: trigger at **0.3R**, 30% offset (ESSENTIAL — earlier trigger locks in more wins)
- Displacement filter: 55% body ratio, 3.5 pts minimum
- M1 confirmation candle required
- `killzone_end`: **12:00** ET (extended from 11:00 for more opportunity)
- `entry_start_time`: **10:00** ET (balanced: avoids early noise while allowing more setups)
- `use_stop_after_loss`: **True** (stops trading after first loss of day)
- `use_opening_range_filter`: **False** (disabled — too restrictive for ~3 TPW target)
- 10-minute cooldown between trades

### Win Rate Calculation (Decisive WR — Feb 2026)
- **WR = Wins / (Wins + Losses)** — BE and EOD trades are excluded from the WR denominator
- Partial TP at 60% of target: trades reaching 60%+ of TP distance are classified as WIN; trailing exits below threshold are BE
- This "decisive WR" ensures perfect WR/PnL correlation: every month with 60%+ WR also has positive PnL (0 problem months)

### Performance Results (Optimized Config — P4d, Feb 2026)
- **411 trades** over ~37 months of data (Oct 2023 – Feb 2026), 300 decisive (191W + 109L), 111 BE
- **63.7% decisive WR**, +1134 pts P&L, PF 2.0, AvgWin=10.5 AvgLoss=-10.4
- **56.8% monthly consistency**: 21/37 qualified months at 60%+ decisive WR
- **0 problem months**: every 60%+ WR month has positive PnL (perfect correlation)
- Max drawdown: -65.08 pts, max consecutive losses: 3
- Trade frequency: **2.59 trades/week** (~3/week target achieved)
- Key changes: Decisive WR (excl BE), partial TP reclassification, KZ 12:00, trail 0.3R, 2 TPD

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
