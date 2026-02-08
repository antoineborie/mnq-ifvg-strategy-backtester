# replit.md

## Overview

This is an **MNQ Futures IFVG Strategy Backtester** — a financial backtesting application for testing an Inversed Fair Value Gap (IFVG) trading strategy on Micro E-mini Nasdaq (MNQ) futures contracts. The application provides a Streamlit-based web UI for configuring strategy parameters, running backtests against historical tick/minute data, and visualizing results with Plotly charts.

The strategy focuses on the US market "killzone" session (9:30–11:00 AM Eastern Time) and identifies fair value gaps (FVGs) that get inversed as trade entry signals, with configurable risk/reward management including break-even triggers.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Application Layer
- **Framework**: Streamlit serves as the web application framework, providing the interactive UI for parameter configuration and results display.
- **Entry point**: `app.py` is the main Streamlit application. `main.py` exists but is a basic placeholder and not the primary entry.
- **Rationale**: Streamlit was chosen for rapid prototyping of data-heavy financial dashboards without needing a separate frontend/backend architecture.

### Core Components

1. **`app.py`** — Streamlit UI layer. Handles page config, custom CSS styling, file selection, and rendering of backtest results. Uses Plotly for interactive charting.

2. **`data_loader.py`** — Data ingestion module. Loads historical MNQ futures data from `.pkl` (pickle) files stored in a `data/` directory. Includes compatibility shims (`PlaceholderStringArray`, `CompatStringDtype`, `_fake_pyx_unpickle`) to handle deserialization of pickle files created with different pandas/Cython versions. Also provides `get_active_contract` for futures contract roll management and `list_data_files` for discovering available data files.

3. **`ifvg_strategy.py`** — The core trading strategy engine (`IFVGStrategy` class). Implements:
   - Fair Value Gap detection with configurable minimum size and max age
   - Killzone-based session filtering (default 9:30–11:00 ET)
   - Risk/reward targeting with configurable ratios
   - Break-even management triggered at a specified RR level
   - Per-day trade limits and cooldown periods
   - Displacement filtering for entry quality
   - Day-by-day processing with timezone conversion (UTC → America/New_York)

4. **`attached_assets/`** — Reference/legacy files showing earlier iterations of the data loading and market context analysis logic. These include `DataMaster` class with ICT (Inner Circle Trader) level calculations (midnight open, previous day high/low, macro highs/lows). These files inform the intended direction of the project but may not be fully integrated yet.

### Data Architecture
- **Storage format**: Pandas DataFrames serialized as `.pkl` (pickle) files, organized by contract and month (e.g., `MNQ_2025_12.pkl`).
- **Data location**: Expected in a `data/` directory at the project root.
- **Index**: DateTime index in UTC, converted to Eastern Time for strategy processing.
- **Columns**: Expected to include at minimum `open`, `high`, `low`, `close`, `volume`, and `symbol` (for contract identification).
- **No database**: The project uses flat file storage only. There is no SQL database or ORM involved.

### Strategy Configuration
The strategy uses a dictionary-based configuration pattern with sensible defaults:
- `min_fvg_size`: 4.0 points minimum gap size
- `max_fvg_age`: 30 bars maximum gap age
- `rr_target`: 2.5 risk-to-reward ratio
- `max_risk_pts` / `min_risk_pts`: 50.0 / 5.0 points risk bounds
- `max_trades_per_day`: 3
- `contract_value`: $2.00 per point (MNQ specification)
- Break-even trigger at 1.5 RR
- 5-minute cooldown between trades

### Project Status
The project is fully implemented with a multi-timeframe IFVG backtest engine and interactive Streamlit dashboard. All core features are complete:
- Multi-TF approach: H1 bias → M15 FVG detection → M15 inversion → M1 retracement entry
- Market structure analysis: daily/weekly/monthly highs/lows, swing points as liquidity references
- Economic calendar module (econ_calendar.py) with hardcoded event dates (Jan-May 2025) and impact analysis
- Target mode selection: Fixed R:R vs SSL/BSL (liquidity level targeting)
- Performance: 28 trades over 5 months, 35.7% WR, PF 2.64, +291 pts profit
- Dashboard with 5 tabs: Equity Curve, Trade Log, Statistics, Daily Analysis, Economic Calendar

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