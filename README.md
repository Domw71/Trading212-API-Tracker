# Trading212 Portfolio Pro v4.3

**A real-time Trading 212 portfolio dashboard with accurate net gain history, persistent tracking, smart warnings, and a polished modern UI.**

Trading212 Portfolio Pro is a lightweight, privacy-focused desktop application that pulls **live positions and cash balances** from the Trading 212 **Public (read-only) API**, then combines that data with your imported **transaction CSV history** to calculate **true total return and net gain** over time.

---

## ✨ Features

### 📊 Portfolio & Performance
- Live positions and cash balance via Trading 212 Public API (v0)
- Accurate **Total Return / Net Gain** calculation  
  *(Current assets − total deposits; requires CSV import)*
- Dashboard cards:
  - Portfolio Value
  - Cash Available (with %)
  - Total Return (color-coded)
  - TTM Dividends
  - Realised P/L
  - Fees Paid

### 📈 Net Gain History
- Interactive **Net Gain History** time-series chart
- Persistent JSON storage (survives app restarts)
- Time range filters:
  - 1h, 4h, 8h, 16h, 24h
  - 1d, 1w, 1m, 3m
  - YTD, 1Y, All Time
- Fixed hover tooltips showing **exact timestamp and £ value**
- Automatic cleanup (max 500 stored history points)

### 📋 Positions & Transactions
- Positions tab:
  - Quantity, Avg Price, Current Price
  - Value, Cost, Unrealised P/L
  - % of total portfolio
- Transactions tab:
  - Searchable, color-coded table
    - Buys (green)
    - Sells (red)
    - Dividends (yellow)
  - Totals summary row

### 📉 Historical Extremes
- Tracks per-ticker:
  - Historical minimum price
  - Historical maximum price
  - Current price
  - % distance from highs/lows

### ⚠️ Smart Warnings
- Stale data warning (>10 minutes old)
- High single-position concentration warning (>25%)
- Tiny negative unrealised P/L values (< £0.05) auto-rounded to £0.00

### 🎨 UI & UX
- Modern dark interface
  - Uses `ttkbootstrap` **darkly** theme if installed
  - Clean fallback theme otherwise
- Automatic refresh every ~60 seconds
- Built-in rate-limit handling (429 errors with countdown timer)

---

## 📋 Requirements

- **Python 3.8+**
- Core dependencies:
  ```bash
  pip install pandas requests matplotlib
