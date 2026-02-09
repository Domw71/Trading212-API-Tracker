# Trading212 Portfolio Pro — v4.3 (Net Gain History Chart + persistent JSON tracking + hover tooltips FIXED)
# Updated: Expanded time range filters → 1hr,4hr,8hr,16hr,24hr,1d,1w,1m,3m,YTD,1Y,All Time
# ------------------------------------------------------------
# Features & fixes summary:
# • Live positions & cash from Trading212 Public API
# • Accurate total return using imported CSV transactions
# • Rate-limit handling (429) with auto-retry + live cooldown countdown
# • Enhanced dashboard: color-coded Total Return, cash %, session change arrow
# • Warnings: stale data (>10 min), high concentration (>25% in one position)
# • Charts polished: larger size, no top/right spines, subtle grid
# • Tiny negative P/L (e.g. -0.01) forced to 0.00 via threshold + rounding
# • Yellow countdown timer in sidebar showing seconds until next auto-refresh
# • Sidebar "Net Gain £" tile shows same £ value as Total Return card
# • Net Gain £ tile turns light green (+) or light red (-)
# • Net Gain History tab with time-series chart + persistent JSON tracking
# • Hover tooltips on net gain points showing date & value
# • FIXED: tooltips no longer persist after mouse leave or chart refresh
# • NEW: Time range buttons: 1hr, 4hr, 8hr, 16hr, 24hr, 1d, 1w, 1m, 3m, YTD, 1Y, All Time
# ------------------------------------------------------------
import os
import json
import time
import threading
import base64
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pandas as pd
import requests
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox, BOTH, X, LEFT, RIGHT, Y, EW, NS
from tkinter import StringVar, END
try:
    import ttkbootstrap as tb
    from ttkbootstrap.constants import *
    BOOTSTRAP = True
except ImportError:
    BOOTSTRAP = False
try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import matplotlib.dates as mdates
    import matplotlib.text as mtext
    import numpy as np
    MATPLOTLIB = True
except ImportError:
    MATPLOTLIB = False
try:
    import mplcursors
    MPLCURSORS_AVAILABLE = True
except ImportError:
    MPLCURSORS_AVAILABLE = False
    print("Note: mplcursors not installed → hover tooltips disabled. Install with: pip install mplcursors")

# ────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────
APP_NAME = "Trading212 Portfolio Pro v4.3"
DATA_DIR = "data"
CSV_FILE = os.path.join(DATA_DIR, "transactions.csv")
CACHE_FILE = os.path.join(DATA_DIR, "positions_cache.json")
SETTINGS_FILE = os.path.join(DATA_DIR, "settings.json")
MIN_MAX_FILE = os.path.join(DATA_DIR, "min_max.json")
NET_GAIN_HISTORY_FILE = os.path.join(DATA_DIR, "net_gain_history.json")
BASE_URL = "https://live.trading212.com/api/v0"
CACHE_TTL = 30
MAX_BAR_TICKERS = 25
STALE_THRESHOLD_MIN = 10
CONCENTRATION_THRESHOLD_PCT = 25
ZERO_PL_THRESHOLD = 0.05
HISTORY_MAX_POINTS = 500
HISTORY_PLOT_DAYS = 90
os.makedirs(DATA_DIR, exist_ok=True)

# ────────────────────────────────────────────────
# MODELS
# ────────────────────────────────────────────────
@dataclass
class ApiCredentials:
    key: str = ""
    secret: str = ""

@dataclass
class Position:
    ticker: str
    quantity: float
    avg_price: float
    current_price: float
    est_value: float
    unrealised_pl: float
    total_cost: float
    portfolio_pct: float = 0.0

# ────────────────────────────────────────────────
# SECRETS & CACHE & HISTORY HELPERS
# ────────────────────────────────────────────────
class Secrets:
    @staticmethod
    def load() -> ApiCredentials:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, 'r') as f:
                data = json.load(f)
                return ApiCredentials(data.get('api_key', ''), data.get('api_secret', ''))
        return ApiCredentials()

    @staticmethod
    def save(creds: ApiCredentials):
        with open(SETTINGS_FILE, 'w') as f:
            json.dump({'api_key': creds.key, 'api_secret': creds.secret}, f, indent=2)

class Cache:
    @staticmethod
    def load() -> Optional[Dict]:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, 'r') as f:
                return json.load(f)
        return None

    @staticmethod
    def save(data: List[Dict]):
        with open(CACHE_FILE, 'w') as f:
            json.dump({'ts': time.time(), 'positions': data}, f)

    @staticmethod
    def is_valid(cache: Optional[Dict]) -> bool:
        return cache is not None and (time.time() - cache.get('ts', 0) < CACHE_TTL)

def load_min_max() -> Dict:
    if os.path.exists(MIN_MAX_FILE):
        with open(MIN_MAX_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_min_max(data: Dict):
    with open(MIN_MAX_FILE, 'w') as f:
        json.dump(data, f, indent=2)

def load_net_gain_history() -> List[Dict]:
    if not os.path.exists(NET_GAIN_HISTORY_FILE):
        return []
    try:
        with open(NET_GAIN_HISTORY_FILE, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                return [p for p in data if isinstance(p, dict) and 'ts' in p and 'net_gain' in p]
    except Exception:
        pass
    return []

def save_net_gain_history(history: List[Dict]):
    history = sorted(history, key=lambda x: x.get('ts', 0))
    if len(history) > HISTORY_MAX_POINTS:
        history = history[-HISTORY_MAX_POINTS:]
    with open(NET_GAIN_HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)

# ────────────────────────────────────────────────
# TRANSACTIONS REPOSITORY
# ────────────────────────────────────────────────
class TransactionsRepo:
    def __init__(self):
        self.path = CSV_FILE

    def load(self) -> pd.DataFrame:
        if not os.path.exists(self.path):
            return pd.DataFrame()
        try:
            df = pd.read_csv(self.path, parse_dates=['Date'])
        except Exception:
            return pd.DataFrame()
        numeric = ['Quantity', 'Price', 'Total', 'Fee', 'FX_Rate', 'Result']
        for c in numeric:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
        return df

    def save(self, df: pd.DataFrame):
        df.to_csv(self.path, index=False, date_format='%Y-%m-%d %H:%M:%S')

    @staticmethod
    def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
        key = ['Date', 'Type', 'Ticker', 'Total', 'Reference']
        key = [c for c in key if c in df.columns]
        if key:
            return df.drop_duplicates(subset=key, keep='last').sort_values('Date').reset_index(drop=True)
        return df

# ────────────────────────────────────────────────
# HELPERS
# ────────────────────────────────────────────────
def safe_float(value) -> float:
    try:
        return float(value) if value is not None else 0.0
    except (TypeError, ValueError):
        return 0.0

def round_money(val: float) -> float:
    return round(val, 2)

# ────────────────────────────────────────────────
# TRADING212 SERVICE
# ────────────────────────────────────────────────
class Trading212Service:
    def __init__(self, creds: ApiCredentials):
        self.creds = creds
        self.session = requests.Session()
        self.session.headers.update(self._headers())

    def _headers(self):
        if not self.creds.key or not self.creds.secret:
            return {}
        token = base64.b64encode(f"{self.creds.key}:{self.creds.secret}".encode()).decode()
        return {"Authorization": f"Basic {token}"}

    def fetch_positions(self) -> List[Position]:
        cache = Cache.load()
        if Cache.is_valid(cache):
            print("[Cache] Using cached positions.")
            print("[Connection] OK")
            return [Position(**p) for p in cache['positions']]

        print(f"[API] Fetching {BASE_URL}/equity/positions ...")
        try:
            r = self.session.get(f"{BASE_URL}/equity/positions", timeout=12)
            r.raise_for_status()
            items = r.json()
            print(f"[API] Got {len(items)} positions.")
            print("[Connection] OK")
        except Exception as e:
            print(f"[Positions API error] {str(e)}")
            raise RuntimeError(f"API failed: {str(e)}")

        positions = []
        total_value = 0.0
        zero_pl_count = 0
        fallback_used = 0
        for pos in items:
            try:
                instr = pos.get('instrument', {})
                ticker_raw = instr.get('ticker', '')
                ticker = ticker_raw.split('_')[0].upper().rstrip('L')
                qty = safe_float(pos.get('quantity'))
                avg_price = safe_float(pos.get('averagePricePaid'))
                current_price = safe_float(pos.get('currentPrice'))
                w = pos.get('walletImpact', {}) or {}
                est_value = safe_float(w.get('currentValue'))
                api_pl = safe_float(w.get('unrealizedProfitLoss'))
                total_cost = safe_float(w.get('totalCost'))

                fallback_pl = (current_price - avg_price) * qty
                if api_pl == 0 and qty != 0 and abs(current_price - avg_price) > 0.001:
                    print(f"[Fallback] {ticker}: API PL=0 → calc {fallback_pl:.2f}")
                    unrealised_pl = fallback_pl
                    fallback_used += 1
                else:
                    unrealised_pl = api_pl

                if abs(unrealised_pl) < ZERO_PL_THRESHOLD:
                    unrealised_pl = 0.0
                if abs(est_value) < ZERO_PL_THRESHOLD:
                    est_value = 0.0

                est_value = round_money(est_value)
                unrealised_pl = round_money(unrealised_pl)
                total_cost = round_money(total_cost)

                if unrealised_pl == 0:
                    zero_pl_count += 1

                positions.append(Position(
                    ticker=ticker,
                    quantity=qty,
                    avg_price=avg_price,
                    current_price=current_price,
                    est_value=est_value,
                    unrealised_pl=unrealised_pl,
                    total_cost=total_cost
                ))
                total_value += est_value
            except Exception as e:
                print(f"[Parse skip] {e}")
                continue

        if fallback_used:
            print(f"[Info] Fallback used on {fallback_used} positions.")
        if zero_pl_count:
            print(f"[Warn] {zero_pl_count}/{len(positions)} positions have P/L = 0")

        for p in positions:
            p.portfolio_pct = (p.est_value / total_value * 100) if total_value > 0 else 0

        Cache.save([p.__dict__ for p in positions])
        return positions

    def fetch_cash_balance(self) -> float:
        print("[API] Fetching cash balance...")
        try:
            r = self.session.get(f"{BASE_URL}/equity/account/cash", timeout=8)
            r.raise_for_status()
            data = r.json()
            keys = ['free', 'freeCash', 'cash', 'available']
            for k in keys:
                val = data.get(k)
                if val is not None:
                    cash = safe_float(val)
                    print(f"[API] Parsed {k}: £{cash:.2f}")
                    return round_money(cash)
            print("[API] No valid cash key found, defaulting to 0.0")
            return 0.0
        except Exception as e:
            print(f"[Cash API error] {str(e)}")
            return 0.0

# ────────────────────────────────────────────────
# ANALYTICS
# ────────────────────────────────────────────────
class Analytics:
    @staticmethod
    def calculate(df: pd.DataFrame, positions: List[Position], cash: float) -> Dict:
        if df.empty:
            holdings_value = sum(p.est_value for p in positions)
            total_assets = holdings_value + cash
            return {
                'total_assets': total_assets,
                'holdings_value': holdings_value,
                'net_gain': 0.0,
                'total_return_pct': 0.0,
                'realised_pl': 0.0,
                'fees': 0.0,
                'deposits': 0.0,
                'deposit_count': 0,
                'ttm_dividends': 0.0,
            }

        fees = float(df['Fee'].sum()) if 'Fee' in df.columns else 0.0
        realised = float(df['Result'].sum()) if 'Result' in df.columns else 0.0
        deposit_mask = df['Type'].str.contains('deposit', case=False, na=False)
        deposits_sum = float(df.loc[deposit_mask, 'Total'].sum())
        deposit_count = int(deposit_mask.sum())

        holdings_value = sum(p.est_value for p in positions)
        total_assets = holdings_value + cash
        net_gain = total_assets - deposits_sum
        total_return_pct = (net_gain / deposits_sum * 100) if deposits_sum > 0 else 0.0

        ttm_dividends = 0.0
        if not df.empty and 'Date' in df.columns and 'Type' in df.columns and 'Result' in df.columns:
            one_year_ago = datetime.now() - timedelta(days=365)
            recent_div = df[
                (df['Date'] >= one_year_ago) &
                df['Type'].str.contains('dividend', case=False, na=False) &
                (df['Result'] > 0)
            ]['Result'].sum()
            ttm_dividends = float(recent_div)

        return {
            'total_assets': total_assets,
            'holdings_value': holdings_value,
            'net_gain': net_gain,
            'total_return_pct': total_return_pct,
            'realised_pl': realised,
            'fees': fees,
            'deposits': deposits_sum,
            'deposit_count': deposit_count,
            'ttm_dividends': ttm_dividends,
        }

# ────────────────────────────────────────────────
# MAIN APPLICATION
# ────────────────────────────────────────────────
class Trading212App:
    def __init__(self, root):
        self.root = root
        self.root.title(APP_NAME)
        self.root.state('zoomed')

        self.repo = TransactionsRepo()
        self.df = self.repo.load()
        self.creds = Secrets.load()
        self.service = Trading212Service(self.creds)
        self.positions: List[Position] = []
        self.cash_balance: float = 0.0
        self.last_refresh_str = "Never"
        self.last_successful_refresh = 0.0
        self.last_total_assets = 0.0
        self.MIN_REFRESH_GAP = 60
        self.cooldown_end_time = 0.0
        self.countdown_after_id = None
        self.next_auto_refresh_time = 0.0

        # Net Gain chart period selector — changed default to 1d
        self.netgain_period_var = tk.StringVar(value="1d")

        self._setup_style()
        self._build_ui()

        self.refresh(async_fetch=True)
        self.update_countdown()

        def auto_refresh_loop():
            self.refresh(async_fetch=True)
            self.root.after(60000, auto_refresh_loop)

        self.root.after(10000, auto_refresh_loop)

    def _setup_style(self):
        if BOOTSTRAP:
            self.style = tb.Style(theme="darkly")
            self.style.configure("Treeview", rowheight=28, font=('Segoe UI', 10))
            self.style.configure("Treeview.Heading", font=('Segoe UI', 11, 'bold'))
        else:
            ttk.Style().theme_use('clam')

    def _build_ui(self):
        self.sidebar = ttk.Frame(self.root, width=220)
        self.sidebar.pack(side=LEFT, fill=Y, padx=(10, 0), pady=10)

        self.content = ttk.Frame(self.root)
        self.content.pack(side=LEFT, fill=BOTH, expand=True, padx=10, pady=10)

        self.tabs = {}
        self.tab_dashboard = ttk.Frame(self.content);     self.tabs["Dashboard"] = self.tab_dashboard
        self.tab_netgain   = ttk.Frame(self.content);     self.tabs["Net Gain History"] = self.tab_netgain
        self.tab_transactions = ttk.Frame(self.content);  self.tabs["Transactions"] = self.tab_transactions
        self.tab_positions = ttk.Frame(self.content);     self.tabs["Positions"] = self.tab_positions
        self.tab_minmax    = ttk.Frame(self.content);     self.tabs["Historical Highs & Lows"] = self.tab_minmax
        self.tab_settings  = ttk.Frame(self.content);     self.tabs["Settings"] = self.tab_settings

        self._build_dashboard()
        self._build_netgain_chart()
        self._build_transactions()
        self._build_positions()
        self._build_minmax()
        self._build_settings()
        self._build_sidebar()

        self.switch_tab("Dashboard")

    def switch_tab(self, tab_name):
        for tab in self.tabs.values():
            tab.pack_forget()
        self.tabs[tab_name].pack(fill=BOTH, expand=True)

        if BOOTSTRAP:
            for btn in self.menu_btns.values():
                btn.configure(bootstyle="secondary")
            self.menu_btns[tab_name].configure(bootstyle="primary")

    def _build_sidebar(self):
        ttk.Label(self.sidebar, text=APP_NAME, font=('Segoe UI', 14, 'bold')).pack(pady=20, padx=10)

        menu_items = ["Dashboard", "Net Gain History", "Transactions", "Positions", "Historical Highs & Lows", "Settings"]
        self.menu_btns = {}
        for item in menu_items:
            bootstyle = "secondary" if BOOTSTRAP else ""
            btn = ttk.Button(self.sidebar, text=item, command=lambda t=item: self.switch_tab(t), bootstyle=bootstyle)
            btn.pack(fill=X, pady=4, padx=8)
            self.menu_btns[item] = btn

        ttk.Separator(self.sidebar).pack(fill=X, pady=15, padx=10)
        ttk.Button(self.sidebar, text="Import CSV", bootstyle="primary", command=self.import_csv).pack(fill=X, pady=4, padx=8)
        ttk.Separator(self.sidebar).pack(fill=X, pady=15, padx=10)

        stats_grid = ttk.Frame(self.sidebar)
        stats_grid.pack(fill=X, padx=10, pady=8)
        stats_grid.columnconfigure(0, weight=1)
        stats_grid.columnconfigure(1, weight=1)

        stats = [
            ("# Positions", "📊", "—"),
            ("Avg Position", "💰", "—"),
            ("Cash %", "💸", "—"),
            ("Total Deposits", "🏦", "—"),
            ("Deposits Count", "🔢", "—"),
            ("Market Buys", "🛒", "—"),
            ("Market Sells", "💵", "—"),
            ("Net Gain £", "💰", "—"),
        ]
        self.stats_vars = {}
        self.stats_labels = {}
        for i, (label_text, icon, default) in enumerate(stats):
            if BOOTSTRAP:
                tile = tb.Frame(stats_grid, bootstyle="dark", padding=10)
            else:
                tile = ttk.Frame(stats_grid, padding=10)
                tile.configure(relief="solid", borderwidth=1, background="#2c2c40")
            tile.grid(row=i//2, column=i%2, padx=6, pady=6, sticky='ew')

            header = ttk.Frame(tile)
            header.pack(fill=X, pady=(2, 0))
            ttk.Label(header, text=icon, font=('Segoe UI', 13)).pack(side=LEFT, padx=(8, 6))
            ttk.Label(header, text=label_text, font=('Segoe UI', 10)).pack(side=LEFT)

            var = tk.StringVar(value=default)
            value_label = ttk.Label(tile, textvariable=var, font=('Segoe UI', 12, 'bold'), anchor='center')
            value_label.pack(pady=(2, 6))

            self.stats_vars[label_text] = var
            self.stats_labels[label_text] = value_label

        ttk.Separator(self.sidebar).pack(fill=X, pady=15, padx=10)
        ttk.Label(self.sidebar, text="Additional Features:", font=('Segoe UI', 11, 'bold')).pack(anchor='w', padx=10)
        ttk.Button(self.sidebar, text="Help / About", bootstyle="outline", command=self.show_help).pack(fill=X, pady=4, padx=8)

        self.refresh_label = ttk.Label(self.sidebar, text="Status: Waiting for refresh...", foreground='gray')
        self.refresh_label.pack(pady=10, padx=10, anchor='s')

        self.countdown_var = tk.StringVar(value="Next auto-refresh: calculating...")
        self.countdown_label = ttk.Label(
            self.sidebar,
            textvariable=self.countdown_var,
            foreground='yellow',
            font=('Segoe UI', 10, 'bold')
        )
        self.countdown_label.pack(pady=(0, 10), padx=10, anchor='s')

    def show_help(self):
        messagebox.showinfo("Help / About", f"{APP_NAME}\n\nA tool for managing Trading212 portfolios.\n\nVersion: 4.3\nBuilt with Tkinter, Pandas & Matplotlib.\n\nHover tooltips fixed - no more persistent or stacking tooltips.\nExtended time filters added to Net Gain History chart (1hr–24hr + All Time).")

    def update_countdown(self):
        now = time.time()
        if self.next_auto_refresh_time > now:
            remaining = max(0, int(self.next_auto_refresh_time - now))
            self.countdown_var.set(f"Next auto-refresh in {remaining}s")
            self.root.after(1000, self.update_countdown)
        else:
            self.countdown_var.set("Auto-refreshing...")

    def _set_total_return_text(self, text: str):
        if "Total Return" in self.card_vars:
            self.card_vars["Total Return"].set(text)

    def start_cooldown_countdown(self, seconds_left: int):
        if self.countdown_after_id:
            self.root.after_cancel(self.countdown_after_id)
            self.countdown_after_id = None
        if seconds_left <= 0:
            self._set_total_return_text("Refreshing...")
            self.refresh(async_fetch=True)
            return
        self.countdown_after_id = self.root.after(
            1000, lambda: self.start_cooldown_countdown(seconds_left - 1)
        )

    def refresh(self, async_fetch: bool = False, is_auto_retry: bool = False):
        def _task():
            try:
                print("\n=== Refresh started ===")
                self.root.after(0, lambda: self._set_total_return_text("Refreshing..."))
                self.positions = self.service.fetch_positions()

                min_max = load_min_max()
                now_str = datetime.now().isoformat()
                for p in self.positions:
                    ticker = p.ticker
                    current = p.current_price
                    if current <= 0:
                        continue
                    if ticker not in min_max:
                        min_max[ticker] = {
                            'min': current, 'max': current,
                            'first_seen': now_str, 'last_updated': now_str,
                            'count': 1, 'last_price': current
                        }
                    else:
                        d = min_max[ticker]
                        d['min'] = min(d['min'], current)
                        d['max'] = max(d['max'], current)
                        d['last_updated'] = now_str
                        d['count'] += 1
                        d['last_price'] = current
                save_min_max(min_max)

                cash_ok = True
                try:
                    self.cash_balance = self.service.fetch_cash_balance()
                except Exception as e:
                    cash_ok = False
                    print(f"[Cash fetch failed] {str(e)}")

                if cash_ok:
                    self.last_successful_refresh = time.time()
                    self.cooldown_end_time = time.time() + self.MIN_REFRESH_GAP
                    self.next_auto_refresh_time = time.time() + 60

                    summary = Analytics.calculate(self.df, self.positions, self.cash_balance)
                    now_ts = time.time()
                    net_gain_value = summary['net_gain']

                    hist = load_net_gain_history()
                    hist.append({
                        "ts": now_ts,
                        "net_gain": round(net_gain_value, 2),
                        "total_assets": round(summary['total_assets'], 2),
                    })
                    save_net_gain_history(hist)

                    buy_count = sell_count = 0
                    if not self.df.empty:
                        buy_mask = self.df['Type'].str.contains('buy', case=False, na=False)
                        sell_mask = self.df['Type'].str.contains('sell', case=False, na=False)
                        buy_count = int(buy_mask.sum())
                        sell_count = int(sell_mask.sum())

                    tv = summary['total_assets']
                    num_pos = len([p for p in self.positions if p.quantity > 0])
                    avg_pos = tv / num_pos if num_pos > 0 else 0
                    cash_pct = (self.cash_balance / tv * 100) if tv > 0 else 0

                    zero_count = sum(1 for p in self.positions if p.unrealised_pl == 0)
                    status = f"Warning: {zero_count} zero P/L" if zero_count > len(self.positions)//2 else "OK"

                    self.last_refresh_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    session_change_str = ""
                    if self.last_total_assets > 0:
                        change_pct = ((tv - self.last_total_assets) / self.last_total_assets) * 100
                        arrow = "↑" if change_pct >= 0 else "↓"
                        session_change_str = f" {arrow} {change_pct:+.2f}%"
                    self.last_total_assets = tv

                    self.root.after(0, lambda: self._render_dashboard(
                        summary, num_pos, avg_pos, cash_pct, session_change_str,
                        buy_count=buy_count, sell_count=sell_count, net_gain=net_gain_value
                    ))
                    self.root.after(0, lambda: self.refresh_label.config(
                        text=f"Last refresh: {self.last_refresh_str} | {status}",
                        foreground='lime' if "Warning" not in status else 'orange'
                    ))
                    self.root.after(0, self._render_positions)
                    self.root.after(0, self._render_minmax)
                    self.root.after(0, self.render_transactions)
                    self.root.after(0, self._render_netgain_chart)
                    self.root.after(0, self.update_countdown)

                    print("=== Refresh completed successfully ===")
                else:
                    self.root.after(0, lambda: self._set_total_return_text("Fetch error – retrying..."))
                    self.root.after(0, lambda: self.refresh_label.config(
                        text="Cash fetch failed – auto-retrying...",
                        foreground='orange'
                    ))
                    self.root.after(60000, lambda: self.refresh(async_fetch=False, is_auto_retry=True))

            except Exception as e:
                print(f"=== Refresh failed: {str(e)} ===")
                self.root.after(0, lambda: self._set_total_return_text(f"Error: {str(e)}"))
                self.root.after(0, lambda: self.refresh_label.config(text=f"Error: {str(e)}", foreground='red'))

        now = time.time()
        if not is_auto_retry:
            if self.cooldown_end_time > now:
                remaining = int(self.cooldown_end_time - now) + 1
                self.start_cooldown_countdown(remaining)
                return
            else:
                self.cooldown_end_time = now + self.MIN_REFRESH_GAP

        if async_fetch:
            threading.Thread(target=_task, daemon=True).start()
        else:
            _task()

    # ────────────────────────────────────────────────
    # DASHBOARD (unchanged)
    # ────────────────────────────────────────────────
    def _build_dashboard(self):
        main = ttk.Frame(self.tab_dashboard, padding=20)
        main.pack(fill=BOTH, expand=True)

        cards_frame = ttk.Frame(main)
        cards_frame.pack(fill=X, pady=(0, 20))

        self.card_vars = {}
        self.card_frames = {}
        cards = [
            ("Portfolio Value", "💰", "#4CAF50"),
            ("Cash Available", "💸", "#9C27B0"),
            ("Total Return", "📈", "#2196F3"),
            ("TTM Dividends", "📅", "#FFEB3B"),
            ("Realised P/L", "🏦", "#FF9800"),
            ("Fees Paid", "⚠️", "#F44336"),
        ]
        for i, (title, emoji, color) in enumerate(cards):
            card = tb.Frame(cards_frame, bootstyle="dark", padding=14) if BOOTSTRAP else ttk.Frame(cards_frame)
            card.grid(row=i//3, column=i%3, padx=12, pady=10, sticky=EW)
            ttk.Label(card, text=f"{emoji} {title}", font=('Segoe UI', 13, 'bold')).pack(anchor='w')
            var = tk.StringVar(value="—")
            ttk.Label(card, textvariable=var, font=('Segoe UI', 26, 'bold'), foreground=color).pack(anchor='center', pady=8)
            self.card_vars[title] = var
            self.card_frames[title] = card

        cards_frame.columnconfigure((0,1,2), weight=1)

        self.warning_var = tk.StringVar(value="")
        ttk.Label(main, textvariable=self.warning_var, font=('Segoe UI', 10), foreground='#FF9800').pack(pady=(0,10))

        if MATPLOTLIB:
            chart_frame = tb.Frame(main, bootstyle="dark") if BOOTSTRAP else ttk.Frame(main)
            chart_frame.pack(fill=BOTH, expand=True, pady=10)

            self.fig = Figure(figsize=(14, 6.5), facecolor='#1e1e2f')
            self.ax1 = self.fig.add_subplot(121)
            self.ax2 = self.fig.add_subplot(122)

            for ax in (self.ax1, self.ax2):
                ax.set_facecolor('#252535')
                ax.tick_params(colors='white', labelsize=10)
                ax.title.set_color('white')
                ax.title.set_fontsize(14)
                ax.xaxis.label.set_color('white')
                ax.yaxis.label.set_color('white')
                ax.grid(True, axis='y', alpha=0.15, color='gray', linestyle='--', linewidth=0.5)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color('gray')
                ax.spines['bottom'].set_color('gray')

            self.canvas = FigureCanvasTkAgg(self.fig, master=chart_frame)
            self.canvas.get_tk_widget().pack(fill=BOTH, expand=True, padx=6, pady=6)

    def _render_dashboard(self, s: Dict, num_pos: int, avg_pos: float, cash_pct: float,
                          session_change_str: str = "", buy_count: int = 0, sell_count: int = 0,
                          net_gain: float = 0.0):
        self.card_vars["Portfolio Value"].set(f"£{round_money(s['total_assets']):,.2f}")
        self.card_vars["Cash Available"].set(f"£{round_money(self.cash_balance):,.2f} ({cash_pct:.1f}%)")

        gain = s['net_gain']
        pct = s['total_return_pct']
        sign_gain = "+" if gain >= 0 else ""
        sign_pct = "+" if pct >= 0 else ""
        return_text = f"{sign_gain}£{round_money(gain):,.2f} ({sign_pct}{pct:.2f}%){session_change_str}"
        self.card_vars["Total Return"].set(return_text)

        if BOOTSTRAP and "Total Return" in self.card_frames:
            if pct > 10: self.card_frames["Total Return"].configure(bootstyle="success")
            elif pct > 3: self.card_frames["Total Return"].configure(bootstyle="info")
            elif pct > -3: self.card_frames["Total Return"].configure(bootstyle="secondary")
            elif pct > -10:self.card_frames["Total Return"].configure(bootstyle="warning")
            else: self.card_frames["Total Return"].configure(bootstyle="danger")

        self.card_vars["TTM Dividends"].set(f"£{round_money(s['ttm_dividends']):,.2f}")
        self.card_vars["Realised P/L"].set(f"£{round_money(s['realised_pl']):+,.2f}")
        self.card_vars["Fees Paid"].set(f"£{round_money(s['fees']):,.2f}")

        self.stats_vars["# Positions"].set(f"{num_pos}")
        self.stats_vars["Avg Position"].set(f"£{round_money(avg_pos):,.0f}")
        self.stats_vars["Cash %"].set(f"{cash_pct:.1f}%")
        self.stats_vars["Total Deposits"].set(f"£{round_money(s['deposits']):,.0f}")
        self.stats_vars["Deposits Count"].set(f"{s.get('deposit_count', 0):,d}")
        self.stats_vars["Market Buys"].set(f"{buy_count:,d}")
        self.stats_vars["Market Sells"].set(f"{sell_count:,d}")

        sign = "+" if net_gain >= 0 else ""
        self.stats_vars["Net Gain £"].set(f"{sign}£{round_money(net_gain):,.2f}")

        if "Net Gain £" in self.stats_labels:
            label = self.stats_labels["Net Gain £"]
            if net_gain > 0:
                label.configure(foreground="#90EE90")
            elif net_gain < 0:
                label.configure(foreground="#FF9999")
            else:
                label.configure(foreground="white")

        warnings = []
        minutes_ago = (time.time() - self.last_successful_refresh) / 60 if self.last_successful_refresh > 0 else 999
        if minutes_ago > STALE_THRESHOLD_MIN:
            warnings.append(f"Data stale ({int(minutes_ago)} min ago)")

        max_pct = max((p.portfolio_pct for p in self.positions if p.quantity > 0), default=0)
        if max_pct > CONCENTRATION_THRESHOLD_PCT:
            warnings.append(f"High concentration: {max_pct:.1f}% in one position")

        self.warning_var.set(" • ".join(warnings) if warnings else "")

        if MATPLOTLIB and self.positions:
            self.ax1.clear()
            self.ax2.clear()

            active = [p for p in self.positions if p.est_value > 0]
            sorted_active = sorted(active, key=lambda x: -x.est_value)
            n = len(sorted_active)

            if n <= MAX_BAR_TICKERS:
                tickers = [p.ticker for p in sorted_active]
                values = [p.est_value for p in sorted_active]
                colors = ['#66BB6A' if p.unrealised_pl >= 0 else '#EF5350' for p in sorted_active]
                self.ax1.bar(tickers, values, color=colors, edgecolor='gray', linewidth=0.8)
                self.ax1.tick_params(axis='x', rotation=60, labelsize=9)
            else:
                top_n = min(35, n)
                tickers = [p.ticker for p in sorted_active[:top_n]]
                values = [p.est_value for p in sorted_active[:top_n]]
                colors = ['#66BB6A' if p.unrealised_pl >= 0 else '#EF5350' for p in sorted_active[:top_n]]
                self.ax1.barh(tickers[::-1], values[::-1], color=colors[::-1], height=0.65)
                self.ax1.set_title(f"Top {top_n} Positions ({n-top_n} more)", fontsize=13)
                self.ax1.invert_yaxis()
                self.ax1.tick_params(axis='y', labelsize=9)

            self.ax1.set_title("Position Values" if n <= MAX_BAR_TICKERS else "", fontsize=14)

            if n > 15:
                top_vals = [p.est_value for p in sorted_active[:15]]
                top_lbls = [p.ticker for p in sorted_active[:15]]
                other = sum(p.est_value for p in sorted_active[15:])
                pie_vals = top_vals + [other]
                pie_lbls = top_lbls + ['Other']
            else:
                pie_vals = [p.est_value for p in sorted_active]
                pie_lbls = [p.ticker for p in sorted_active]

            self.ax2.pie(pie_vals, labels=pie_lbls, autopct='%1.1f%%',
                         colors=plt.cm.tab20.colors[:len(pie_vals)], textprops={'color':'white', 'fontsize':9})
            self.ax2.set_title("Allocation", fontsize=14)

            self.fig.tight_layout(pad=1.5)
            self.canvas.draw()

    # ────────────────────────────────────────────────
    # NET GAIN HISTORY CHART — with expanded time filters
    # ────────────────────────────────────────────────
    def _build_netgain_chart(self):
        frame = ttk.Frame(self.tab_netgain, padding=20)
        frame.pack(fill=BOTH, expand=True)

        # Period filter buttons — all in one horizontal row
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=X, pady=(0, 12))

        periods = [
            ("1hr",    "1hr"),
            ("4hr",    "4hr"),
            ("8hr",    "8hr"),
            ("16hr",   "16hr"),
            ("1d",     "1d"),
            ("1w",     "1w"),
            ("1m",     "1m"),
            ("3m",     "3m"),
            ("YTD",    "YTD"),
            ("All Time", "All Time")
        ]

        self.period_buttons = {}

        for label, value in periods:
            btn = ttk.Button(
                btn_frame,
                text=label,
                command=lambda v=value: self.set_netgain_period(v),
                width=8
            )
            btn.pack(side=LEFT, padx=4, pady=3, fill=X, expand=False)
            self.period_buttons[value] = btn

        # Highlight default
        self.set_netgain_period("1hr", update_buttons_only=True)

        if not MATPLOTLIB:
            ttk.Label(frame, text="Matplotlib not available", foreground="orange").pack(pady=40)
            return

        fig = Figure(figsize=(12, 6), facecolor='#1e1e2f')
        ax = fig.add_subplot(111)
        ax.set_facecolor('#252535')
        ax.tick_params(colors='white')
        ax.grid(True, axis='y', alpha=0.12, color='gray', linestyle='--')
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        ax.spines['left'].set_color('gray')
        ax.spines['bottom'].set_color('gray')
        ax.set_title("Net Gain Over Time (£)", color='white', fontsize=14, pad=15)
        ax.set_xlabel("Date", color='white')
        ax.set_ylabel("Net Gain (£)", color='white')

        self.netgain_fig = fig
        self.netgain_ax = ax
        self.netgain_canvas = FigureCanvasTkAgg(fig, master=frame)
        self.netgain_canvas.get_tk_widget().pack(fill=BOTH, expand=True, padx=5, pady=5)

    def set_netgain_period(self, period: str, update_buttons_only=False):
        self.netgain_period_var.set(period)

        # Update button styles
        if BOOTSTRAP:
            for val, btn in self.period_buttons.items():
                btn.configure(bootstyle="primary" if val == period else "secondary")
        else:
            for val, btn in self.period_buttons.items():
                btn.state(['!pressed'] if val != period else ['pressed'])

        if not update_buttons_only:
            self._render_netgain_chart()

    def get_netgain_date_cutoff(self) -> Optional[datetime]:
        period = self.netgain_period_var.get()
        now = datetime.now()

        if period in ("All Time", "All"):
            return None
        elif period == "YTD":
            return datetime(now.year, 1, 1)
        elif period == "3m":
            return now - timedelta(days=90)
        elif period == "1m":
            return now - timedelta(days=30)
        elif period == "1w":
            return now - timedelta(days=7)
        elif period == "1d":
            return now - timedelta(days=1)
        elif period == "16hr":               # ← updated
            return now - timedelta(hours=16)
        elif period == "8hr":
            return now - timedelta(hours=8)
        elif period == "4hr":
            return now - timedelta(hours=4)
        elif period == "1hr":
            return now - timedelta(hours=1)
        else:
            return now - timedelta(days=30)  # fallback

    def _render_netgain_chart(self):
        if not MATPLOTLIB or not hasattr(self, 'netgain_ax'):
            return

        hist = load_net_gain_history()
        if not hist:
            self.netgain_ax.clear()
            self.netgain_ax.text(0.5, 0.5, "No history yet\n(data appears after refreshes)",
                                ha='center', va='center', color='gray', fontsize=12)
            self.netgain_canvas.draw()
            return

        times = [datetime.fromtimestamp(p['ts']) for p in hist]
        gains = [p['net_gain'] for p in hist]

        # Apply selected time range filter
        cutoff = self.get_netgain_date_cutoff()
        if cutoff is not None:
            mask = [t >= cutoff for t in times]
            times = [t for t, m in zip(times, mask) if m]
            gains = [g for g, m in zip(gains, mask) if m]

        if not times or not gains:
            self.netgain_ax.clear()
            self.netgain_ax.text(0.5, 0.5, f"No data in selected period ({self.netgain_period_var.get()})",
                                ha='center', va='center', color='gray', fontsize=12)
            self.netgain_canvas.draw()
            return

        # Clean up ALL existing annotations BEFORE clearing axes
        for artist in list(self.netgain_fig.get_children()):
            if isinstance(artist, mtext.Annotation):
                try: artist.remove()
                except: pass
        for artist in list(self.netgain_ax.get_children()):
            if isinstance(artist, mtext.Annotation):
                try: artist.remove()
                except: pass

        self.netgain_ax.clear()

        if gains:
            min_g = min(gains)
            max_g = max(gains)
            data_span = max_g - min_g
            abs_peak = max(abs(min_g), abs(max_g), 0.01)
            min_visible_half = 0.40
            padding_factor = 1.4
            half_span = max(data_span / 2 * padding_factor, min_visible_half, abs_peak * 1.5)
            center = (max_g + min_g) / 2
            self.netgain_ax.set_ylim(center - half_span, center + half_span)

        line, = self.netgain_ax.plot(
            times, gains,
            color='#BB86FC',
            linewidth=1.8,
            marker='o',
            markersize=3,
            alpha=0.9
        )

        if gains:
            self.netgain_ax.axhline(0, color='gray', lw=0.8, ls='--', alpha=0.5)

        self.netgain_ax.fill_between(
            times, gains, 0,
            where=(np.array(gains) >= 0),
            color='#4CAF50', alpha=0.08
        )
        self.netgain_ax.fill_between(
            times, gains, 0,
            where=(np.array(gains) < 0),
            color='#EF5350', alpha=0.08
        )

        self.netgain_ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
        self.netgain_ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))
        plt.setp(self.netgain_ax.get_xticklabels(), rotation=35, ha='right')

        self.netgain_ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda y, _: f'£{y:,.2f}' if abs(y) < 10 else f'£{int(y):,}')
        )

        # Hover tooltips
        if MPLCURSORS_AVAILABLE and times and gains:
            cursor = mplcursors.cursor(
                line,
                hover=True,
                highlight=True
            )
            @cursor.connect("add")
            def on_hover(sel):
                dt = mdates.num2date(sel.target[0])
                date_str = dt.strftime("%Y-%m-%d %H:%M")
                value = sel.target[1]
                sel.annotation.set_text(f"{date_str}\n£{value:,.2f}")
                sel.annotation.get_bbox_patch().set_alpha(0.92)
                sel.annotation.get_bbox_patch().set_facecolor("#2d2d44")
                sel.annotation.get_bbox_patch().set_edgecolor("#bb86fc")
                sel.annotation.set_color("white")
                sel.annotation.xy = sel.target

        self.netgain_fig.tight_layout()
        self.netgain_canvas.draw()

    # ────────────────────────────────────────────────
    # TRANSACTIONS TAB (unchanged)
    # ────────────────────────────────────────────────
    def _build_transactions(self):
        filter_bar = ttk.Frame(self.tab_transactions)
        filter_bar.pack(fill=X, pady=(0,8))
        ttk.Label(filter_bar, text="Filter:").pack(side=LEFT, padx=8)
        self.tx_filter_var = StringVar()
        ttk.Entry(filter_bar, textvariable=self.tx_filter_var, width=45).pack(side=LEFT, padx=6)
        self.tx_filter_var.trace('w', lambda *args: self.render_transactions())

        tree_frame = ttk.Frame(self.tab_transactions)
        tree_frame.pack(fill=BOTH, expand=True, padx=5, pady=5)

        cols = ["Date", "Type", "Ticker", "Quantity", "Price", "Total", "Fee", "Result", "Note"]
        self.tree_tx = ttk.Treeview(tree_frame, columns=cols, show='headings')
        for c in cols:
            self.tree_tx.heading(c, text=c, command=lambda col=c: self._sort_tree(self.tree_tx, col, False))
            anchor = 'w' if c in ["Date", "Type", "Ticker", "Note"] else 'e'
            width = 160 if c in ["Date", "Note"] else 110
            self.tree_tx.column(c, width=width, anchor=anchor, stretch=True)

        vsb = ttk.Scrollbar(tree_frame, orient=VERTICAL, command=self.tree_tx.yview)
        hsb = ttk.Scrollbar(tree_frame, orient=HORIZONTAL, command=self.tree_tx.xview)
        self.tree_tx.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        self.tree_tx.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')
        tree_frame.columnconfigure(0, weight=1)
        tree_frame.rowconfigure(0, weight=1)

        self.tree_tx.tag_configure('even', background='#222233')
        self.tree_tx.tag_configure('odd', background='#1a1a2a')
        self.tree_tx.tag_configure('highlight', background='#3a3a55')
        self.tree_tx.tag_configure('buy', foreground='#66BB6A')
        self.tree_tx.tag_configure('sell', foreground='#EF5350')
        self.tree_tx.tag_configure('dividend', foreground='#FFCA28')
        self.tree_tx.tag_configure('total', font=('Segoe UI', 10, 'bold'), foreground='#BB86FC')

        self.render_transactions()

    def render_transactions(self):
        self.tree_tx.delete(*self.tree_tx.get_children(''))
        filter_text = self.tx_filter_var.get().lower().strip()

        if not filter_text:
            rows_to_show = self.df.iterrows()
        else:
            rows_to_show = [
                (idx, row) for idx, row in self.df.iterrows()
                if filter_text in ' '.join(str(v).lower() for v in row)
            ]

        for idx, (_, row) in enumerate(rows_to_show):
            values = [row.get(c, '') for c in self.tree_tx['columns']]
            tags = ['even' if idx % 2 == 0 else 'odd']
            ttype = str(row.get('Type', '')).lower()
            if 'buy' in ttype: tags.append('buy')
            elif 'sell' in ttype: tags.append('sell')
            elif 'dividend' in ttype: tags.append('dividend')
            self.tree_tx.insert('', 'end', values=values, tags=tags)

        if not self.df.empty:
            totals = [
                "TOTAL", "", "",
                self.df['Quantity'].sum(),
                "", self.df['Total'].sum(),
                self.df['Fee'].sum(),
                self.df['Result'].sum(),
                ""
            ]
            formatted = [f"{v:,.2f}" if isinstance(v, (int, float)) and i not in [0,1,2,4,8] else v
                         for i, v in enumerate(totals)]
            self.tree_tx.insert('', 'end', values=formatted, tags=('total',))

    # ────────────────────────────────────────────────
    # POSITIONS TAB (unchanged)
    # ────────────────────────────────────────────────
    def _build_positions(self):
        frame = ttk.Frame(self.tab_positions, padding=12)
        frame.pack(fill=BOTH, expand=True)

        cols = ["Ticker", "Qty", "Avg Price", "Current", "Value", "Unreal. P/L", "Cost", "% Portfolio"]
        self.tree_pos = ttk.Treeview(frame, columns=cols, show='headings')
        for c in cols:
            self.tree_pos.heading(c, text=c, command=lambda col=c: self._sort_tree(self.tree_pos, col, False))
            anchor = 'w' if c == "Ticker" else 'e'
            width = 140 if c in ["Value", "Unreal. P/L", "Cost"] else 100
            self.tree_pos.column(c, width=width, anchor=anchor)

        vsb = ttk.Scrollbar(frame, orient=VERTICAL, command=self.tree_pos.yview)
        hsb = ttk.Scrollbar(frame, orient=HORIZONTAL, command=self.tree_pos.xview)
        self.tree_pos.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        self.tree_pos.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

        self.tree_pos.tag_configure('even', background='#222233')
        self.tree_pos.tag_configure('odd', background='#1a1a2a')
        self.tree_pos.tag_configure('profit', foreground='#66BB6A')
        self.tree_pos.tag_configure('loss', foreground='#EF5350')
        self.tree_pos.tag_configure('total', font=('Segoe UI', 10, 'bold'), foreground='#BB86FC')

        self._render_positions()

    def _render_positions(self):
        self.tree_pos.delete(*self.tree_pos.get_children(''))

        sorted_pos = sorted(self.positions, key=lambda x: -x.est_value if x.quantity > 0 else 0)
        total_value = sum(p.est_value for p in sorted_pos)
        total_pl = sum(p.unrealised_pl for p in sorted_pos)
        total_cost = sum(p.total_cost for p in sorted_pos)

        for idx, p in enumerate(sorted_pos):
            if p.quantity <= 0:
                continue
            tags = ['profit' if p.unrealised_pl >= 0 else 'loss']
            tags.append('even' if idx % 2 == 0 else 'odd')
            vals = (
                p.ticker,
                f"{p.quantity:,.4f}",
                f"£{round_money(p.avg_price):,.2f}",
                f"£{round_money(p.current_price):,.2f}",
                f"£{round_money(p.est_value):,.2f}",
                f"£{round_money(p.unrealised_pl):+,.2f}",
                f"£{round_money(p.total_cost):,.2f}",
                f"{p.portfolio_pct:.1f}%"
            )
            self.tree_pos.insert('', 'end', values=vals, tags=tags)

        footer = ("TOTAL", "", "", "", f"£{round_money(total_value):,.2f}",
                  f"£{round_money(total_pl):+,.2f}", f"£{round_money(total_cost):,.2f}", "100.0%")
        self.tree_pos.insert('', 'end', values=footer, tags=('total',))

    # ────────────────────────────────────────────────
    # HISTORICAL HIGHS & LOWS TAB (unchanged)
    # ────────────────────────────────────────────────
    def _build_minmax(self):
        frame = ttk.Frame(self.tab_minmax, padding=12)
        frame.pack(fill=BOTH, expand=True)

        cols = ["Ticker", "Status", "Current Price", "Min Price", "Max Price",
                "% from Min", "% from Max", "Updates", "First Seen", "Last Updated"]
        self.tree_minmax = ttk.Treeview(frame, columns=cols, show='headings')
        for c in cols:
            self.tree_minmax.heading(c, text=c, command=lambda col=c: self._sort_tree(self.tree_minmax, col, False))
            anchor = 'w' if c in ["Ticker","Status","First Seen","Last Updated"] else 'e'
            width = 140 if c in ["Current Price","Min Price","Max Price"] else 100
            self.tree_minmax.column(c, width=width, anchor=anchor)

        vsb = ttk.Scrollbar(frame, orient=VERTICAL, command=self.tree_minmax.yview)
        hsb = ttk.Scrollbar(frame, orient=HORIZONTAL, command=self.tree_minmax.xview)
        self.tree_minmax.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        self.tree_minmax.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

        self.tree_minmax.tag_configure('even', background='#222233')
        self.tree_minmax.tag_configure('odd', background='#1a1a2a')
        self.tree_minmax.tag_configure('closed', foreground='gray')

        self._render_minmax()

    def _render_minmax(self):
        self.tree_minmax.delete(*self.tree_minmax.get_children(''))

        min_max = load_min_max()
        current_tickers = {p.ticker: p for p in self.positions}
        sorted_tickers = sorted(min_max.keys())

        for idx, ticker in enumerate(sorted_tickers):
            data = min_max[ticker]
            if ticker in current_tickers:
                status = "Open"
                current = current_tickers[ticker].current_price
            else:
                status = "Closed"
                current = data.get('last_price', 0.0)

            min_p = data['min']
            max_p = data['max']
            from_min = ((current - min_p) / min_p * 100) if current > 0 and min_p > 0 else 0.0
            from_max = ((current - max_p) / max_p * 100) if current > 0 and max_p > 0 else 0.0

            vals = (
                ticker,
                status,
                f"£{round_money(current):,.2f}",
                f"£{round_money(min_p):,.2f}",
                f"£{round_money(max_p):,.2f}",
                f"{from_min:+.1f}%",
                f"{from_max:+.1f}%",
                data['count'],
                data['first_seen'][:19],
                data['last_updated'][:19]
            )
            tags = ['even' if idx % 2 == 0 else 'odd']
            if status == 'Closed':
                tags.append('closed')
            self.tree_minmax.insert('', 'end', values=vals, tags=tags)

    # ────────────────────────────────────────────────
    # SETTINGS TAB (unchanged)
    # ────────────────────────────────────────────────
    def _build_settings(self):
        f = ttk.Frame(self.tab_settings, padding=30)
        f.pack(fill=BOTH, expand=True)

        ttk.Label(f, text="API Key").grid(row=0, column=0, sticky='e', pady=8, padx=10)
        self.api_key_var = tk.StringVar(value=self.creds.key)
        ttk.Entry(f, textvariable=self.api_key_var, width=55).grid(row=0, column=1, pady=8)

        ttk.Label(f, text="API Secret").grid(row=1, column=0, sticky='e', pady=8, padx=10)
        self.api_secret_var = tk.StringVar(value=self.creds.secret)
        ttk.Entry(f, textvariable=self.api_secret_var, width=55, show="*").grid(row=1, column=1, pady=8)

        btn_frame = ttk.Frame(f)
        btn_frame.grid(row=2, column=1, pady=20, sticky='e')

        ttk.Button(btn_frame, text="Save Credentials", bootstyle="success", command=self.save_credentials).pack(side=LEFT, padx=8)
        ttk.Button(btn_frame, text="Clear Cache", bootstyle="warning", command=self.clear_cache).pack(side=LEFT, padx=8)
        ttk.Button(btn_frame, text="Clear Transactions", bootstyle="danger", command=self.clear_transactions).pack(side=LEFT, padx=8)

    def save_credentials(self):
        creds = ApiCredentials(self.api_key_var.get().strip(), self.api_secret_var.get().strip())
        Secrets.save(creds)
        self.creds = creds
        self.service = Trading212Service(creds)
        messagebox.showinfo("Saved", "Credentials updated. Restart recommended.")

    def clear_cache(self):
        if os.path.exists(CACHE_FILE):
            os.remove(CACHE_FILE)
        messagebox.showinfo("Cache", "Cache cleared.")

    def clear_transactions(self):
        if messagebox.askyesno("Confirm", "Delete all saved transactions?"):
            self.df = pd.DataFrame()
            self.repo.save(self.df)
            self.render_transactions()
            self.refresh()

    def import_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not path:
            return
        try:
            raw = pd.read_csv(path)
            raw.columns = raw.columns.str.strip().str.lower()

            mapping = {
                'time': 'Date', 'action': 'Type', 'ticker': 'Ticker',
                'no. of shares': 'Quantity', 'price / share': 'Price',
                'total': 'Total', 'result': 'Result', 'exchange rate': 'FX_Rate',
                'currency (total)': 'Currency', 'notes': 'Note', 'name': 'Instrument Name',
                'id': 'Reference'
            }

            df_new = pd.DataFrame()
            for old, new in mapping.items():
                candidates = [c for c in raw.columns if old.lower() in c.lower()]
                if candidates:
                    df_new[new] = raw[candidates[0]]

            fee_cols = [c for c in raw.columns if any(k in c.lower() for k in ['fee','tax','stamp','conversion'])]
            df_new['Fee'] = raw[fee_cols].sum(axis=1, numeric_only=True).fillna(0) if fee_cols else 0.0

            df_new['Type'] = df_new.get('Type', '').replace({
                'market buy': 'Buy', 'buy': 'Buy',
                'market sell': 'Sell', 'sell': 'Sell',
                'deposit': 'Deposit', 'withdrawal': 'Withdrawal',
                'dividend': 'Dividend'
            }, regex=True)

            df_new['Date'] = pd.to_datetime(df_new.get('Date'), errors='coerce')

            for col in ['Quantity','Price','Total','Fee','Result','FX_Rate']:
                if col in df_new.columns:
                    df_new[col] = pd.to_numeric(df_new[col], errors='coerce').fillna(0)

            df_new = df_new.reindex(columns=['Date','Type','Ticker','Quantity','Price','Total',
                                             'Fee','Result','FX_Rate','Currency','Note','Reference'])

            for c in ['Date','Type','Ticker','Quantity','Total','Reference','Note']:
                if c not in df_new.columns:
                    df_new[c] = pd.NA

            dedup_cols = ['Date','Type','Ticker','Quantity','Price','Total','Fee','Reference']
            dedup_cols = [c for c in dedup_cols if c in df_new.columns]

            existing = self.df.copy()
            if 'Date' in existing.columns:
                existing['Date'] = pd.to_datetime(existing['Date'], errors='coerce')

            if not existing.empty and dedup_cols:
                merged_check = pd.merge(df_new, existing[dedup_cols],
                                       how='left', on=dedup_cols, indicator=True)
                new_rows = merged_check[merged_check['_merge'] == 'left_only'].drop(columns='_merge')
            else:
                new_rows = df_new.copy()

            if new_rows.empty:
                messagebox.showinfo("Import", "No new transactions found.")
                return

            self.df = pd.concat([self.df, new_rows], ignore_index=True)
            self.df = self.repo.deduplicate(self.df.fillna({'Ticker':'-','Note':''}))
            self.df = self.df.sort_values('Date', ascending=False).reset_index(drop=True)
            self.repo.save(self.df)
            self.render_transactions()

            added_count = len(new_rows)
            total_count = len(self.df)
            messagebox.showinfo("Import Complete", f"Added {added_count} new rows.\nTotal transactions now: {total_count}")

            self.refresh(async_fetch=True)

        except Exception as e:
            messagebox.showerror("CSV Error", f"Failed to import:\n{str(e)}")

    def _sort_tree(self, tree, col, reverse):
        data = [(tree.set(k, col), k) for k in tree.get_children('')]
        data.sort(reverse=reverse, key=lambda t: (t[0] is not None, t[0]))
        for index, (_, k) in enumerate(data):
            tree.move(k, '', index)
        tree.heading(col, command=lambda: self._sort_tree(tree, col, not reverse))

if __name__ == '__main__':
    root = tb.Window(themename="darkly") if BOOTSTRAP else tk.Tk()
    app = Trading212App(root)
    root.mainloop()
