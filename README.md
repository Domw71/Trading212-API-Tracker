- The app opens a window with tabs: Dashboard, Transactions, Positions, AI Analyst, Settings.

## Setup API Credentials (for Live Positions & Cash)

Trading212 provides a Public API (Live) for account data like positions and cash balance.

### Step-by-Step: Generate API Key & Secret

1. Open the Trading212 mobile or web app.
2. Go to **Settings** > **API (Live)**.
3. Accept the mandatory risk warning.
4. Tap **Generate API key**.
5. Fill the form:
- Give it a name (e.g., "Portfolio Pro").
- Choose IP restriction (recommended: restrict to your IP for security; or unrestricted if testing).
6. Submit — you will see:
- **API Key** (visible anytime)
- **API Secret** (shown **only once** — copy it immediately!)
7. If you lose the secret, generate a new key pair (old one revoked).

**Security Tips**:
- Treat secret like password — never share.
- Revoke keys anytime in Settings > API (Live).
- API rate limits apply (e.g. ~1 req/sec for positions; check headers like `x-ratelimit-remaining`).

### Enter Credentials in App

1. Go to **Settings** tab.
2. Paste **API Key** and **API Secret** into the fields.
3. Click **Save Credentials**.
4. Restart app (recommended for changes to take effect).

Now the app fetches live positions and cash on refresh.

## Importing CSV Transactions from Trading212

The app calculates total return using deposits/withdrawals from your transaction history (CSV) + live holdings/cash.

### Export CSV from Trading212

1. In Trading212 app/web: Go to **Account** > **History**.
2. Tap the **Export** button (or three dots > Export).
3. Select timeframe (e.g. all available) and data types (orders, transactions, dividends, etc.).
4. Download the .csv file.

**Limitations**:
- Not available for CFD accounts.
- Export may not include all columns every time (depends on recent activity).

**Typical Columns** (based on Trading212 format):
- Time / Date
- Action / Type (e.g. Buy, Sell, Deposit, Dividend)
- Ticker
- No. of shares / Quantity
- Price / share
- Total
- Result (P/L for sells/dividends)
- Exchange rate / FX Rate
- Currency
- Notes
- Name / Instrument Name
- ID / Reference

### Import into App

1. In app sidebar: Click **Import CSV**.
2. Select your exported .csv file.
3. App auto-maps columns (e.g. "Time" → Date, "Action" → Type, etc.).
4. Deduplicates and sorts transactions.
5. Shows success message with added rows count.
6. Refresh dashboard — total return now uses your full history.

**Tip**: Import periodically (e.g. monthly) for accurate returns. App saves transactions to `data/transactions.csv`.

## How to Use the App

1. **Refresh Data** (sidebar button or auto on start):
- Fetches live positions & cash via API.
- If rate-limited (429), shows "Fetch error – retrying in 60s..." + auto-retries.
- Quick manual clicks → live countdown + auto-refresh at 0.

2. **Tabs Overview**:
- **Dashboard**: Key metrics cards (Portfolio Value, Cash, Total Return, etc.) + charts (positions bar/pie).
  - Color-coded Total Return card (green/red based on %).
  - Session change arrow (↑/↓ since last refresh).
  - Warnings for stale data (>10 min) or high concentration (>25% in one position).
- **Transactions**: Table of imported trades/dividends/deposits.
  - Filter by text (ticker, type, etc.).
  - Totals row at bottom.
- **Positions**: Live holdings table (from API).
  - Qty, Avg Price, Current, Value, Unreal. P/L, Cost, % Portfolio.
- **AI Analyst**: Basic insights on positions (profit/loss %, thoughts like "take partial profits").
- **Settings**: API credentials, clear cache/transactions, debug API.

3. **Common Actions**:
- First use: Set API creds → Import CSV → Refresh.
- Monitor: Refresh often (live data updates).
- Analyze: Use AI Analyst for quick thoughts.
- Troubleshoot: Check console logs; Debug API saves response to `data/api_debug.json`.

## Understanding the Data

- **Portfolio Value**: Sum of all positions' current value + cash available (live from API).
- **Cash Available**: Free cash balance (from `/equity/account/cash` endpoint, prefers 'free' key).
- **Total Return %**: `(current assets - total deposits) / total deposits * 100`  
- Deposits/withdrawals from imported CSV (Type contains "deposit").
- Realised P/L + unrealised P/L + dividends contribute.
- **Unrealised P/L**: Current value - cost basis (from API or fallback calc if API returns 0).
- **Realised P/L**: Sum of 'Result' column from sells/dividends in CSV.
- **TTM Dividends**: Dividends (Result >0, Type contains "dividend") in last 365 days.
- **Fees Paid**: Sum of 'Fee' column in CSV.
- **% Portfolio**: Position value / total holdings value (excludes cash).
- **Session Change**: % change in portfolio value since last successful refresh.
- **Warnings**:
- Stale: No refresh >10 min → data outdated.
- Concentration: One position >25% → risk warning.

**Data Sources**:
- Live prices/positions/cash → Trading212 Public API (real-time).
- Historical deposits/sells/dividends/fees → Your CSV imports.

Enjoy tracking your portfolio!  
Report issues or contribute on GitHub (if you host it).

Last updated: January 31, 2026
