import requests
import hashlib
import hmac
import time
import logging
import pandas as pd
import pandas_ta as ta
import os
import numpy as np
import json 
from datetime import datetime, timezone
from apscheduler.schedulers.blocking import BlockingScheduler

# --- Core configuration ---
API_KEY = "qJ7dLk28aZp3VhR9T2mN5yX4cB8sGfQ1wE6rUjD3H0nCzK5PoL4iMb7SYt9Aa2Fx"
SECRET = "zXcV1bN5mQwE3rT7yUiP9oA0sDdF4gJ6hKlZ2xC8vBnM4qW2eRtY6uI1oPaS5dF7g"
BASE_URL = "https://mock-api.roostoo.com"

# --- Data and log files ---
HISTORY_FILE = "price_history_new.csv"
LOG_FILE = "advanced_trading_bot.log"
PORTFOLIO_STATE_FILE = "portfolio_state.json"

# --- Strategy parameters ---
CASH_ALLOCATION = 0.9           
TOP_N = 5                       
MAX_POSITION_RATIO = 0.05       
MIN_ORDER_VALUE = 300           

# --- Filter parameters ---
VOLUME_FILTER_PERCENTILE = 0.7    
VOLATILITY_FILTER_PERCENTILE = 0.7
VOLATILITY_WINDOW = 7           
VOLUME_LOOKBACK_DAYS = 5        

# --- Momentum and Timing Indicators ---
MOMENTUM_LOOKBACK_DAYS = 3      
ENTRY_SMA_WINDOW = 7            
EXIT_SMA_WINDOW = 10           
RSI_WINDOW = 7                  
RSI_ENTRY_MIN = 50              
RSI_ENTRY_MAX = 70              
RSI_EXIT_OVERBOUGHT = 80        
RSI_EXIT_OVERSOLD = 30         

# --- Risk management parameters ---
STOP_LOSS_THRESHOLD = -0.08    
TAKE_PROFIT_LEVELS = {          
    3: {"profit": 0.15, "sell_ratio": 0.30},
    2: {"profit": 0.10, "sell_ratio": 0.40},
    1: {"profit": 0.05, "sell_ratio": 0.30},
}

# --- Log configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

def load_portfolio_state():
    if not os.path.exists(PORTFOLIO_STATE_FILE):
        return {}
    try:
        with open(PORTFOLIO_STATE_FILE, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        logging.error(f"Unable to parse {PORTFOLIO_STATE_FILE}, an empty state will be used.")
        return {}
    except Exception as e:
        logging.error(f"Load {PORTFOLIO_STATE_FILE} failure: {e}")
        return {}

def save_portfolio_state(state):
    try:
        with open(PORTFOLIO_STATE_FILE, 'w') as f:
            json.dump(state, f, indent=4)
    except Exception as e:
        logging.error(f"Save {PORTFOLIO_STATE_FILE} failure: {e}")

# --- API Client (RoostooV3Client) ---
class RoostooV3Client:
    def __init__(self, api_key, secret_key):
        self.api_key = api_key
        self.secret = secret_key
        self.base_url = BASE_URL
        logging.info("Roostoo V3 client initialization completed")

    def _get_timestamp_ms(self):
        return str(int(time.time() * 1000))

    def _generate_signature(self, params):
        query_string = '&'.join(["{}={}".format(k, params[k])
                             for k in sorted(params.keys())])
        us = self.secret.encode('utf-8')
        m = hmac.new(us, query_string.encode('utf-8'), hashlib.sha256)
        return m.hexdigest()

    def _signed_request(self, method, endpoint, params=None):
        if params is None:
            params = {}
        
        params["timestamp"] = self._get_timestamp_ms()
        signature = self._generate_signature(params)
        
        headers = {"RST-API-KEY": self.api_key, "MSG-SIGNATURE": signature}
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method.upper() == 'GET':
                response = requests.get(url, params=params, headers=headers, timeout=15)
            elif method.upper() == 'POST':
                headers['Content-Type'] = 'application/x-www-form-urlencoded'
                response = requests.post(url, data=params, headers=headers, timeout=15)
            else:
                logging.error(f"Unsupported request methods: {method}")
                return None

            response.raise_for_status()
            if endpoint not in ["/v3/balance", "/v3/ticker"]:
                 logging.info(f"API response ({method} {endpoint}): {response.json()}")
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"API {method} request failed ({url}): {e}")
            if e.response: logging.error(f"Response content: {e.response.text}")
            return None

    def get_ticker(self, pair=None):
        url = self.base_url+ "/v3/ticker"
        params = {"timestamp": self._get_timestamp_ms()}
        if pair: params["pair"] = pair
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to retrieve Ticker market data: {e}")
            return None

    def get_balance(self):
        return self._signed_request("GET", "/v3/balance")

    def place_order(self, coin, side, quantity, price=None):
        data = {
            "pair": coin + "/USD",
            "side": side,
            "quantity": quantity,
        }
        if not price:
            data['type'] = "MARKET"
        else:
            data['type'] = "LIMIT"
            data['price'] = price
        
        logging.info(f"Ready to place an order: {data}")
        return self._signed_request("POST", "/v3/place_order", params=data)

    def query_order(self, pending_only="FALSE", pair=None):
        data = {"pending_only": pending_only}
        if pair: data["pair"] = pair
        return self._signed_request("POST", "/v3/query_order", params=data)

# --- Data collection ---
def collect_daily_price_data():
    logging.info("--- Start daily price and volume collection task ---")
    client = RoostooV3Client(API_KEY, SECRET)
    ticker_data = client.get_ticker()
    
    if not ticker_data or not ticker_data.get("Success"):
        logging.error("Failed to retrieve market data; unable to update historical data.")
        return

    today_utc_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    updates = {"date": today_utc_str}
    for pair, data in ticker_data.get("Data", {}).items():
        if pair.endswith("/USD"):
            updates[pair] = data.get("LastPrice")
            updates[f"{pair}_volume"] = data.get("UnitTradeValue")
    
    df = pd.read_csv(HISTORY_FILE) if os.path.exists(HISTORY_FILE) else pd.DataFrame(columns=["date"])
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

    if today_utc_str in df['date'].values:
        idx = df[df['date'] == today_utc_str].index
        for col, val in updates.items():
            if col not in df.columns: df[col] = np.nan
            df.loc[idx, col] = val
    else:
        new_row = pd.DataFrame([updates])
        df = pd.concat([df, new_row], ignore_index=True)

    df.to_csv(HISTORY_FILE, index=False)
    logging.info(f"The historical data file '{HISTORY_FILE}' has been successfully updated(containing prices and trading volumes).")

# --- Advanced strategy analysis module ---
class StrategyAnalytics:
    def __init__(self, history_file, current_ticker_map):
        self.ticker_map = current_ticker_map
        try:
            self.history_df_raw = pd.read_csv(history_file)
            self.history_df_raw['date'] = pd.to_datetime(self.history_df_raw['date'])
            self.history_df_raw.set_index('date', inplace=True)
            
            price_cols = [c for c in self.history_df_raw.columns if not c.endswith('_volume')]
            volume_cols = [c for c in self.history_df_raw.columns if c.endswith('_volume')]
            self.price_df = self.history_df_raw[price_cols].copy()
            self.volume_df = self.history_df_raw[volume_cols].copy()
            
            logging.info(f"The analysis module successfully loaded historical data, totaling {len(self.price_df)} records.")
        except FileNotFoundError:
            self.price_df = None
            self.volume_df = None
            logging.error(f"Historical data file '{history_file}' not found! The policy cannot be executed.")

    def filter_and_rank_assets(self):
        if self.price_df is None: return []

        days_of_data = len(self.price_df)
        
        if days_of_data < VOLUME_LOOKBACK_DAYS:
            logging.info(f"Data is less than {VOLUME_LOOKBACK_DAYS} days ({days_of_data} days), use real-time transaction volume for filtering.")
            volumes = pd.Series({p: d.get('UnitTradeValue', 0) for p, d in self.ticker_map.items() if p.endswith("/USD")})
        else:
            logging.info(f"Data has reached {days_of_data} days, and is filtered using the average transaction volume over the past {VOLUME_LOOKBACK_DAYS} days.")
            avg_volumes = self.volume_df.tail(VOLUME_LOOKBACK_DAYS).mean()
            avg_volumes.index = avg_volumes.index.str.replace('_volume', '')
            volumes = avg_volumes

        volume_threshold = volumes.quantile(VOLUME_FILTER_PERCENTILE)
        volume_candidates = set(volumes[volumes.notna() & (volumes >= volume_threshold)].index)
        logging.info(f"Transaction volume filtering: {len(volume_candidates)}/{len(volumes)} assets meet the requirement (>{volume_threshold:.2f})")

        returns = self.price_df.pct_change(fill_method=None)
        volatility = returns.rolling(window=VOLATILITY_WINDOW).std().iloc[-1]
        volatility.dropna(inplace=True)
        volatility_threshold = volatility.quantile(VOLATILITY_FILTER_PERCENTILE)
        volatility_candidates = set(volatility[volatility >= volatility_threshold].index)
        logging.info(f"Volatility filtering: {len(volatility_candidates)}/{len(volatility)} assets meet the requirement (>{volatility_threshold:.6f})")

        if len(self.price_df) < MOMENTUM_LOOKBACK_DAYS + 1: return []
        momentum = self.price_df.pct_change(MOMENTUM_LOOKBACK_DAYS, fill_method=None).iloc[-1]
        
        final_candidates = list(volume_candidates & volatility_candidates)
        
        ranked_assets = []
        for pair in final_candidates:
            if pair in momentum.index and pd.notna(momentum[pair]):
                ranked_assets.append({
                    "pair": pair,
                    "momentum": momentum[pair],
                    "volatility": volatility.get(pair, 0)
                })

        ranked_assets.sort(key=lambda x: x["momentum"], reverse=True)
        logging.info(f"After filtering and ranking, the top {len(ranked_assets)} candidate assets are selected.")
        return ranked_assets[:TOP_N]

    def check_entry_conditions(self, pair):
        if self.price_df is None or pair not in self.price_df.columns: return False
        if len(self.price_df) < RSI_WINDOW: return False
        
        series = self.price_df[pair].dropna()
        if len(series) < RSI_WINDOW: return False

        sma7 = ta.sma(series, length=ENTRY_SMA_WINDOW).iloc[-1]
        rsi7 = ta.rsi(series, length=RSI_WINDOW).iloc[-1]
        last_price = series.iloc[-1]
        price_above_sma_threshold = last_price > (sma7 * 0.95)
        rsi_ok = RSI_ENTRY_MIN < rsi7 < RSI_ENTRY_MAX
        
        logging.debug(f"Entry check ({pair}): Price={last_price:.2f}, SMA7_Threshold={(sma7 * 0.95):.2f} (>{'T' if price_above_sma_threshold else 'F'}), "
                      f"RSI7={rsi7:.2f} ({RSI_ENTRY_MIN}<RSI<{RSI_ENTRY_MAX} -> {'T' if rsi_ok else 'F'})")
        
        return price_above_sma_threshold and rsi_ok

    def check_exit_conditions(self, pair):
        if self.price_df is None or pair not in self.price_df.columns: return False
        if len(self.price_df) < EXIT_SMA_WINDOW: return False

        series = self.price_df[pair].dropna()
        if len(series) < EXIT_SMA_WINDOW: return False

        sma10 = ta.sma(series, length=EXIT_SMA_WINDOW).iloc[-1]
        last_price = series.iloc[-1]
        if last_price < sma10:
            logging.info(f"Exit signal ({pair}): Price {last_price:.2f} has fallen below the 10-day average {sma10:.2f}.")
            return True

        if len(series) >= RSI_WINDOW:
            rsi7 = ta.rsi(series, length=RSI_WINDOW).iloc[-1]
            if rsi7 > RSI_EXIT_OVERBOUGHT:
                logging.info(f"Exit signal ({pair}): RSI ({rsi7:.2f}) Entering the overbought zone > {RSI_EXIT_OVERBOUGHT}.")
                return True
            if rsi7 < RSI_EXIT_OVERSOLD:
                logging.info(f"Exit signal ({pair}): RSI ({rsi7:.2f}) Entering oversold / panic zone < {RSI_EXIT_OVERSOLD}.")
                return True
        return False

# -- Stop-loss and take-profit actuators ---
def check_sl_tp():
    logging.info("="*10 + " Start performing hourly stop-loss and take-profit checks " + "="*10)
    client = RoostooV3Client(API_KEY, SECRET)
    
    portfolio_state = load_portfolio_state()
    if not portfolio_state:
        logging.info("Status file is empty, skip the stop-loss and take-profit checks. ")
        return
    
    ticker_data = client.get_ticker()
    if not ticker_data or not ticker_data.get("Success"):
        logging.error("Unable to obtain Ticker market data; SL/TP check aborted. ")
        return
    ticker_map = ticker_data.get("Data", {})

    balance_data = client.get_balance()
    if not balance_data or not balance_data.get("Success"):
        logging.error("Unable to retrieve account balance; SL/TP check aborted. ")
        return
    
    wallet = balance_data.get("Wallet") or balance_data.get("SpotWallet") or {}
    current_holdings = {}
    for coin, data in wallet.items():
        if coin == "USD": continue
        pair = f"{coin}/USD"
        quantity = data.get("Free", 0) 
        if quantity > 0:
            current_holdings[pair] = {"quantity": quantity, "coin": coin}

    if not current_holdings:
        logging.info("There are currently no actual positions, so skip the SL/TP logic. ")
        return

    new_state = portfolio_state.copy()
    
    for pair, holding in current_holdings.items():
        coin = holding['coin']
        current_quantity = holding['quantity']

        if pair not in new_state:
            logging.warning(f"Unable to perform SL/TP for {pair} : No purchase record in status file. ")
            continue
            
        if pair not in ticker_map:
            logging.warning(f"nable to perform SL/TP for {pair} : The price for this trading pair is not available in Ticker. ")
            continue

        state_item = new_state[pair]
        avg_buy_price = state_item.get("avg_buy_price", 0)
        original_quantity = state_item.get("original_quantity", 0) 
        tp_stage = state_item.get("tp_stage", 0)
        
        if avg_buy_price <= 0 or original_quantity <= 0:
            logging.warning(f"Skip {pair}: Invalid status file data (price or quantity is 0). ")
            continue

        current_price = ticker_map[pair].get("LastPrice", 0)
        if current_price <= 0: continue

        profit_ratio = (current_price - avg_buy_price) / avg_buy_price
        logging.info(f"Check {pair}: Cost ${avg_buy_price:.4f}, Current Price ${current_price:.4f}, Profit {profit_ratio*100:.2f}% (TP Stage: {tp_stage})")

        if profit_ratio <= STOP_LOSS_THRESHOLD:
            logging.info(f"!!! [STOP-LOSS] !!! {pair} triggers -8% loss, sell all {current_quantity} {coin}")
            client.place_order(coin, "SELL", int(current_quantity))
            del new_state[pair] 
            continue 

        sell_triggered = False
        
        level_15 = TAKE_PROFIT_LEVELS[3]
        if profit_ratio >= level_15["profit"] and tp_stage < 3:
            sell_ratio = 0
            if tp_stage == 0: sell_ratio = level_15["sell_ratio"] + TAKE_PROFIT_LEVELS[2]["sell_ratio"] + TAKE_PROFIT_LEVELS[1]["sell_ratio"]
            elif tp_stage == 1: sell_ratio = level_15["sell_ratio"] + TAKE_PROFIT_LEVELS[2]["sell_ratio"]
            elif tp_stage == 2: sell_ratio = level_15["sell_ratio"]
            
            qty_to_sell = original_quantity * sell_ratio
            logging.info(f"!!! [TAKE-PROFIT 15%] !!! {pair} triggers, sell {sell_ratio*100:.0f}% (quantity: {qty_to_sell})")
            new_state[pair]["tp_stage"] = 3
            sell_triggered = True

        level_10 = TAKE_PROFIT_LEVELS[2]
        if not sell_triggered and profit_ratio >= level_10["profit"] and tp_stage < 2:
            sell_ratio = 0
            if tp_stage == 0: sell_ratio = level_10["sell_ratio"] + TAKE_PROFIT_LEVELS[1]["sell_ratio"]
            elif tp_stage == 1: sell_ratio = level_10["sell_ratio"]
            
            qty_to_sell = original_quantity * sell_ratio
            logging.info(f"!!! [TAKE-PROFIT 10%] !!! {pair} triggers, sell {sell_ratio*100:.0f}% (quantity: {qty_to_sell})")
            new_state[pair]["tp_stage"] = 2
            sell_triggered = True

        level_5 = TAKE_PROFIT_LEVELS[1]
        if not sell_triggered and profit_ratio >= level_5["profit"] and tp_stage < 1:
            sell_ratio = level_5["sell_ratio"]
            
            qty_to_sell = original_quantity * sell_ratio
            logging.info(f"!!! [TAKE-PROFIT 5%] !!! {pair} triggers, sell {sell_ratio*100:.0f}% (quantity: {qty_to_sell})")
            new_state[pair]["tp_stage"] = 1
            sell_triggered = True


        if sell_triggered and qty_to_sell > 0:
            final_sell_qty = min(current_quantity, qty_to_sell)
            logging.info(f"Execute a take-profit sell order {pair}: Planned sell {qty_to_sell:.6f}, Actual available {current_quantity:.6f}, Final sell {final_sell_qty:.6f}")
            if final_sell_qty > 0:
                client.place_order(coin, "SELL", int(final_sell_qty))
                new_state[pair]["original_quantity"] = max(0, original_quantity - final_sell_qty)
                if new_state[pair]["original_quantity"] == 0:
                    logging.info(f"{pair} All items have been sold at a profit and removed from the status. ")
                    del new_state[pair]

    save_portfolio_state(new_state)
    logging.info("="*10 + " Hourly stop-loss and take-profit checks completed " + "="*10)


# --- Main strategy ---
def run_strategy():
    logging.info("="*10 + " Start advanced strategy execution task " + "="*10)
    client = RoostooV3Client(API_KEY, SECRET)
    
    ticker_data = client.get_ticker()
    if not ticker_data or not ticker_data.get("Success"):
        logging.error("Unable to obtain Ticker market data, strategy terminated. ")
        return
    ticker_map = ticker_data.get("Data", {})

    analytics = StrategyAnalytics(HISTORY_FILE, ticker_map)
    top_candidates = analytics.filter_and_rank_assets()
    
    if not top_candidates:
        logging.warning("After filtering, no candidate assets were found. ")
        return

    final_targets = []
    for asset in top_candidates:
        if analytics.check_entry_conditions(asset['pair']):
            final_targets.append(asset)
    
    if not final_targets:
        logging.info("Market condition assessment: Since all top candidate assets do not meet the entry criteria, the market is deemed to be in a downtrend, and trading is suspended. ")
        return
        
    logging.info(f"Final investment targets ({len(final_targets)}): {[t['pair'] for t in final_targets]}")

    balance_data = client.get_balance()
    if not balance_data or not balance_data.get("Success"):
        logging.error("Unable to retrieve account balance, strategy aborted. ")
        return
    

    portfolio_state = load_portfolio_state()

    wallet = balance_data.get("Wallet") or balance_data.get("SpotWallet") or {}
    total_portfolio_value = wallet.get("USD", {}).get("Free", 0)
    current_holdings = {}
    for coin, data in wallet.items():
        if coin == "USD": continue
        pair = f"{coin}/USD"
        if pair in ticker_map and ticker_map[pair].get("LastPrice", 0) > 0:
            price = ticker_map[pair]["LastPrice"]
            quantity = data.get("Free", 0) + data.get("Lock", 0)
            value = price * quantity
            current_holdings[pair] = {"quantity": quantity, "value": value}
            total_portfolio_value += value
    
    logging.info(f"Total asset value (estimated): ${total_portfolio_value:.2f}")

    equity_to_invest = total_portfolio_value * CASH_ALLOCATION
    inverse_volatilities = {t['pair']: 1 / t['volatility'] for t in final_targets if t['volatility'] > 0}
    sum_inverse_vol = sum(inverse_volatilities.values())
    
    target_positions = {}
    if sum_inverse_vol > 0:
        for pair, inv_vol in inverse_volatilities.items():
            weight = inv_vol / sum_inverse_vol
            weight = min(weight, MAX_POSITION_RATIO)
            target_positions[pair] = equity_to_invest * weight
    
    all_pairs_in_play = set(current_holdings.keys()) | set(target_positions.keys())
    
    for pair in all_pairs_in_play:
        target_value = target_positions.get(pair, 0)
        current_value = current_holdings.get(pair, {}).get("value", 0)
        coin = pair.split('/')[0]

        if pair in current_holdings and analytics.check_exit_conditions(pair):
            target_value = 0 
            logging.info(f"Policy exit: {pair} triggered the exit condition (SMA/RSI), and the target value was set to 0. ")

        trade_value = target_value - current_value
        
        if abs(trade_value) < MIN_ORDER_VALUE:
            logging.debug(f"Transaction value too small, skip {pair}: target ${target_value:.2f}, current ${current_value:.2f}")
            continue

        price = ticker_map.get(pair, {}).get("LastPrice")
        if not price or price <= 0: continue

        quantity = abs(trade_value) / price
        
        if trade_value < 0:
            sell_quantity = min(current_holdings[pair]['quantity'], quantity)
            logging.info(f"Rebalancing [Sell]: {pair}, Sell value ${abs(trade_value):.2f} (Quantity: {sell_quantity:.6f})")
            client.place_order(coin, "SELL", int(sell_quantity))

            if pair in portfolio_state:
                portfolio_state[pair]["original_quantity"] = max(0, portfolio_state[pair].get("original_quantity", 0) - sell_quantity)
                if portfolio_state[pair]["original_quantity"] == 0:
                    logging.info(f"After rebalancing and selling, the quantity of {pair} is 0, and it is removed from the status.")
                    del portfolio_state[pair]
            
        else:
         
            logging.info(f"Rebalancing [Buy]: {pair}, Buy value ${trade_value:.2f} (Quantity: {quantity:.6f})")
            client.place_order(coin, "BUY", int(quantity))

            if pair not in portfolio_state:
                portfolio_state[pair] = {
                    "avg_buy_price": price,
                    "original_quantity": quantity,
                    "tp_stage": 0
                }
                logging.info(f"Status Update [New]: {pair} Cost {price}, Quantity {quantity}")
            else:
                old_item = portfolio_state[pair]
                old_quantity = old_item.get("original_quantity", 0)
                old_avg_price = old_item.get("avg_buy_price", 0)
                
                old_value = old_quantity * old_avg_price
                new_value = quantity * price
                new_quantity = old_quantity + quantity
                
                if new_quantity > 0:
                    new_avg_price = (old_value + new_value) / new_quantity
                    portfolio_state[pair]["avg_buy_price"] = new_avg_price
                    portfolio_state[pair]["original_quantity"] = new_quantity
                    portfolio_state[pair]["tp_stage"] = 0
                    logging.info(f"Status Update [Add to Position]: {pair} New cost {new_avg_price:.4f}, New quantity {new_quantity:.6f}, TP Reset")


    save_portfolio_state(portfolio_state)
    
    logging.info("="*10 + " Advanced strategy execution task completed " + "="*10)


if __name__ == "__main__":
    logging.info("Advanced trading robot activated...")
    try:
        collect_daily_price_data()
        run_strategy()
    except Exception as e:
        logging.critical(f"A serious error occurred on first run: {e}", exc_info=True)

    scheduler = BlockingScheduler(timezone="UTC")

    scheduler.add_job(collect_daily_price_data, 'cron', hour='0,4,8,12,16,20', minute=0)

    scheduler.add_job(run_strategy, 'cron', hour='0,4,8,12,16,20', minute=15)
    
    scheduler.add_job(check_sl_tp, 'cron', minute=30)
    
    logging.info("The scheduler has been configured. The robot is entering continuous operation mode...")
    print("="*60)
    print("The advanced trading robot is running. Press Ctrl+C to stop it.")
    print(f"The logs will be recorded in the {LOG_FILE} file.")
    print(f"Scheduling tasks: ")
    print(f"  [Data Collection]: 6 times per day (00:00, 04:00, 08:00, 12:00, 16:00, 20:00 UTC)")
    print(f"  [Strategy Trading]: 6 times per day (00:15, 04:15, 08:15, 12:15, 16:15, 20:15 UTC)")
    print(f"  [Stop-profit/Stop-loss]: Once per hour (at the 30-minute mark of each hour)")
    print("="*60)
    
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logging.info("The trading bot has been manually stopped.")