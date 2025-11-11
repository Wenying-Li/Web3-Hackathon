#!/usr/bin/env python
# -*- coding: utf-8 -*-

import requests
import hashlib
import hmac
import time
import logging
import pandas as pd
import pandas_ta as ta
import os
import numpy as np
import math
from datetime import datetime, timezone
from apscheduler.schedulers.blocking import BlockingScheduler
import json

# --- 核心配置 ---
API_KEY = "qJ7dLk28aZp3VhR9T2mN5yX4cB8sGfQ1wE6rUjD3H0nCzK5PoL4iMb7SYt9Aa2Fx"
SECRET = "zXcV1bN5mQwE3rT7yUiP9oA0sDdF4gJ6hKlZ2xC8vBnM4qW2eRtY6uI1oPaS5dF7g"
BASE_URL = "https://mock-api.roostoo.com"

HISTORY_FILE = "price_history_new.csv"
LOG_FILE = "advanced_trading_bot.log"

CASH_ALLOCATION = 0.9
TOP_N = 5
MAX_POSITION_RATIO = 0.25
MIN_ORDER_VALUE = 300

VOLUME_FILTER_PERCENTILE = 0.7
VOLATILITY_FILTER_PERCENTILE = 0.7
VOLATILITY_WINDOW = 7
VOLUME_LOOKBACK_DAYS = 5

MOMENTUM_LOOKBACK_DAYS = 3
ENTRY_SMA_WINDOW = 7
EXIT_SMA_WINDOW = 10
RSI_WINDOW = 7
RSI_ENTRY_MIN = 50
RSI_ENTRY_MAX = 70
RSI_EXIT_OVERBOUGHT = 80
RSI_EXIT_OVERSOLD = 30

STOP_LOSS_RATIO = 0.95
TAKE_PROFIT_RATIO = 1.33

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

class RoostooV3Client:
    def __init__(self, api_key, secret_key):
        self.api_key = api_key
        self.secret = secret_key
        self.base_url = BASE_URL
        logging.info("Roostoo V3 客户端初始化完成")

    def _get_timestamp_ms(self):
        return str(int(time.time() * 1000))

    def _generate_signature(self, params):
        query_string = '&'.join(["{}={}".format(k, params[k]) for k in sorted(params.keys())])
        us = self.secret.encode('utf-8')
        m = hmac.new(us, query_string.encode('utf-8'), hashlib.sha256)
        return m.hexdigest()
    
    def get_server_time(self):
        r = requests.get(self.base_url + "/v3/serverTime")
        print(r.status_code, r.text)
        return r.json()

    def get_ex_info(self):
        """获取交易所信息, 包含交易对规则。改进：不直接依赖 Success 字段，改为容错解析 symbols"""
        try:
            r = requests.get(self.base_url + "/v3/exchangeInfo", timeout=10)
            r.raise_for_status()
            try:
                res = r.json()
            except Exception:
                logging.error("exchangeInfo 返回非 JSON 内容")
                logging.debug(r.text)
                return None
            # 记录原始响应 debug 级别，便于排查返回格式
            logging.debug("exchangeInfo 原始响应: " + json.dumps(res, default=str))
            logging.info("成功获取交易所信息 (exchangeInfo)。")
            return res
        except requests.exceptions.RequestException as e:
            logging.error(f"获取交易所信息失败: {e}")
            return None
    
    def get_ticker(self, pair=None):
        url = self.base_url+ "/v3/ticker"
        params = {"timestamp": int(time.time())}
        if pair: params["pair"] = pair
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"获取 Ticker 行情失败: {e}")
            return None

    def get_balance(self):
        payload = {
            "timestamp": int(time.time()) * 1000,
        }

        r = requests.get(
            self.base_url + "/v3/balance",
            params=payload,
            headers={"RST-API-KEY": self.api_key,
                    "MSG-SIGNATURE": self._generate_signature(payload)}
        )
        logging.debug(f"get_balance 返回: {r.status_code} {r.text}")    
        try:
            return r.json()
        except Exception:
            return None

    def place_order(self, coin, side, quantity, price=None):
        payload = {
            "timestamp": int(time.time()) * 1000,
            "pair": coin + "/USD",
            "side": side,
            "quantity": quantity,
        }

        if not price:
            payload['type'] = "MARKET"
        else:
            payload['type'] = "LIMIT"
            payload['price'] = price
        
        logging.info(f"准备下单: {payload}")
        r = requests.post(
            self.base_url + "/v3/place_order",
            data=payload,
            headers={"RST-API-KEY": self.api_key,
                     "MSG-SIGNATURE": self._generate_signature(payload)}
        )
        logging.debug(f"place_order 返回: {r.status_code} {r.text}")
        try:
            return r.json()
        except Exception:
            return None

    def query_order(self, pending_only="FALSE", pair=None):
        payload = {
            "timestamp": int(time.time())*1000,
        }
        if pair:
             payload['pair'] = pair
        if pending_only:
             payload['pending_only'] = pending_only

        r = requests.post(
            self.base_url + "/v3/query_order",
            data=payload,
            headers={"RST-API-KEY": self.api_key,
                     "MSG-SIGNATURE": self._generate_signature(payload)}
        )
        logging.debug(f"query_order 返回: {r.status_code} {r.text}")
        try:
            return r.json()
        except Exception:
            return None

# adjust_quantity_by_step 保持不变
def adjust_quantity_by_step(quantity, step_size_str):
    if not step_size_str or float(step_size_str) <= 0:
        return quantity

    step_size = float(step_size_str)
    multiplier = math.floor(quantity / step_size)
    adjusted_quantity = multiplier * step_size
    
    if '.' in str(step_size_str):
        decimal_places = len(str(step_size_str).split('.')[1])
        return float(f"{adjusted_quantity:.{decimal_places}f}")
    else:
        return int(adjusted_quantity)

def collect_daily_price_data():
    logging.info("--- 开始每日价格与交易量采集任务 ---")
    client = RoostooV3Client(API_KEY, SECRET)
    ticker_data = client.get_ticker()
    
    if not ticker_data or not ticker_data.get("Success"):
        logging.error("获取行情失败，无法更新历史数据。")
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
        logging.warning(f"日期 {today_utc_str} 的数据已存在，将进行覆盖。")
        idx = df[df['date'] == today_utc_str].index
        for col, val in updates.items():
            if col not in df.columns: df[col] = np.nan
            df.loc[idx, col] = val
    else:
        new_row = pd.DataFrame([updates])
        df = pd.concat([df, new_row], ignore_index=True)

    df.to_csv(HISTORY_FILE, index=False)
    logging.info(f"历史数据文件 '{HISTORY_FILE}' 已成功更新 (包含价格和交易量)。")

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
            
            logging.info(f"分析模块成功加载历史数据，共 {len(self.price_df)} 条记录。")
        except FileNotFoundError:
            self.price_df = None
            self.volume_df = None
            logging.error(f"历史数据文件 '{history_file}' 未找到！策略无法执行。")

    def filter_and_rank_assets(self):
        if self.price_df is None: return []

        days_of_data = len(self.price_df)
        
        if days_of_data < VOLUME_LOOKBACK_DAYS:
            logging.info(f"数据不足{VOLUME_LOOKBACK_DAYS}天({days_of_data}天)，使用实时交易量进行筛选。")
            volumes = pd.Series({p: d.get('UnitTradeValue', 0) for p, d in self.ticker_map.items() if p.endswith("/USD")})
        else:
            logging.info(f"数据已达{days_of_data}天，使用过去{VOLUME_LOOKBACK_DAYS}天平均交易量进行筛选。")
            avg_volumes = self.volume_df.tail(VOLUME_LOOKBACK_DAYS).mean()
            avg_volumes.index = avg_volumes.index.str.replace('_volume', '')
            volumes = avg_volumes

        volume_threshold = volumes.quantile(VOLUME_FILTER_PERCENTILE)
        volume_candidates = set(volumes[volumes.notna() & (volumes >= volume_threshold)].index)
        logging.info(f"交易量筛选: {len(volume_candidates)}/{len(volumes)} 个资产满足条件 (>{volume_threshold:.2f})")

        returns = self.price_df.pct_change(fill_method=None)
        volatility = returns.rolling(window=VOLATILITY_WINDOW).std().iloc[-1]
        volatility.dropna(inplace=True)
        volatility_threshold = volatility.quantile(VOLATILITY_FILTER_PERCENTILE)
        volatility_candidates = set(volatility[volatility >= volatility_threshold].index)
        logging.info(f"波动率筛选: {len(volatility_candidates)}/{len(volatility)} 个资产满足条件 (>{volatility_threshold:.6f})")

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
        logging.info(f"筛选和排名完成，选出 Top {len(ranked_assets)} 候选资产。")
        return ranked_assets[:TOP_N]

    def check_entry_conditions(self, pair):
        if self.price_df is None or pair not in self.price_df.columns: return False
        if len(self.price_df) < RSI_WINDOW: return False
        
        series = self.price_df[pair].dropna()
        if len(series) < RSI_WINDOW: return False

        sma7 = ta.sma(series, length=ENTRY_SMA_WINDOW).iloc[-1]
        rsi7 = ta.rsi(series, length=RSI_WINDOW).iloc[-1]
        last_price = series.iloc[-1]
        
        price_above_sma = last_price > sma7
        rsi_ok = RSI_ENTRY_MIN < rsi7 < RSI_ENTRY_MAX
        
        logging.debug(f"入场检查 ({pair}): Price={last_price:.2f}, SMA7={sma7:.2f}, RSI7={rsi7:.2f}")
        
        return price_above_sma and rsi_ok

    def check_exit_conditions(self, pair):
        if self.price_df is None or pair not in self.price_df.columns: return False
        if len(self.price_df) < EXIT_SMA_WINDOW: return False

        series = self.price_df[pair].dropna()
        if len(series) < EXIT_SMA_WINDOW: return False

        sma10 = ta.sma(series, length=EXIT_SMA_WINDOW).iloc[-1]
        last_price = series.iloc[-1]
        if last_price < sma10:
            logging.info(f"出场信号 ({pair}): 价格 {last_price:.2f} 已跌破10日均线 {sma10:.2f}。")
            return True

        if len(series) >= RSI_WINDOW:
            rsi7 = ta.rsi(series, length=RSI_WINDOW).iloc[-1]
            if rsi7 > RSI_EXIT_OVERBOUGHT:
                logging.info(f"出场信号 ({pair}): RSI ({rsi7:.2f}) 进入超买区域 > {RSI_EXIT_OVERBOUGHT}。")
                return True
            if rsi7 < RSI_EXIT_OVERSOLD:
                logging.info(f"出场信号 ({pair}): RSI ({rsi7:.2f}) 进入超卖/恐慌区域 < {RSI_EXIT_OVERSOLD}。")
                return True
        return False

# ---
# 主策略逻辑（主要改动：更宽容解析 exchangeInfo）
# ---
def run_strategy():
    logging.info("="*10 + " 开始高级策略执行任务 " + "="*10)
    client = RoostooV3Client(API_KEY, SECRET)

    ex_info = client.get_ex_info()
    if not ex_info:
        logging.error("无法获取交易所信息 (exchangeInfo), 策略中止。")
        return

    # 容错解析 symbols（兼容不同字段名与无 Success 字段的情况）
    symbols_list = None
    if isinstance(ex_info, dict):
        # 可能字段名: 'symbols' 或 'data' / 'Data'
        symbols_list = ex_info.get('symbols') or ex_info.get('data') or ex_info.get('Data')
        # 有些返回会把 symbols 放到更深层或结构不同，尝试根据已知形态额外查找
        if symbols_list is None:
            # 检查常见的可能性
            for k in ex_info.keys():
                if isinstance(ex_info[k], list) and len(ex_info[k]) > 0 and isinstance(ex_info[k][0], dict):
                    # 如果列表项看起来像交易对字典（含 'pair' 字段），则使用它
                    if 'pair' in ex_info[k][0]:
                        symbols_list = ex_info[k]
                        break

    if not symbols_list or not isinstance(symbols_list, list):
        logging.error("exchangeInfo 中未找到可用的 symbols 字段，策略中止。 exchangeInfo 原始内容示例(已记录为 debug)：")
        logging.debug(json.dumps(ex_info, default=str))
        return

    # 将交易对规则存入字典
    symbols_info = {}
    for s in symbols_list:
        # 有些实现可能直接存 pair，有些可能用 symbol/name；尽量兼容
        key = s.get('pair') or s.get('symbol') or s.get('name')
        if not key:
            continue
        symbols_info[key] = s

    ticker_data = client.get_ticker()
    if not ticker_data or not ticker_data.get("Success"):
        logging.error("无法获取Ticker行情, 策略中止。")
        return
    ticker_map = ticker_data.get("Data", {})

    analytics = StrategyAnalytics(HISTORY_FILE, ticker_map)
    top_candidates = analytics.filter_and_rank_assets()
    
    if not top_candidates:
        logging.warning("经过筛选后，没有任何候选资产。")
        return

    final_targets = []
    for asset in top_candidates:
        if analytics.check_entry_conditions(asset['pair']):
            final_targets.append(asset)
    
    if not final_targets:
        logging.info("市场状况判断：所有Top候选资产均不满足入场条件，认定市场为下降趋势，暂停交易。")
        return
        
    logging.info(f"最终投资目标 ({len(final_targets)}个): {[t['pair'] for t in final_targets]}")

    balance_data = client.get_balance()
    if not balance_data or not balance_data.get("Success"):
        logging.error("无法获取账户余额, 策略中止。")
        return
    

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
    
    logging.info(f"总资产价值 (估算): ${total_portfolio_value:.2f}")

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
            logging.info(f"强制退出: {pair} 触发了退出条件，目标价值设为0。")

        trade_value = target_value - current_value
        
        if abs(trade_value) < MIN_ORDER_VALUE:
            logging.debug(f"交易价值过小，跳过 {pair}: 目标 ${target_value:.2f}, 当前 ${current_value:.2f}")
            continue

        price = ticker_map.get(pair, {}).get("LastPrice")
        if not price or price <= 0: continue

        # --- 根据交易规则调整数量精度 ---
        pair_info = symbols_info.get(pair)
        if not pair_info:
            logging.warning(f"无法找到 {pair} 的交易规则，跳过下单。")
            continue
        step_size = pair_info.get('stepSize')

        quantity_unadjusted = abs(trade_value) / price
        quantity = adjust_quantity_by_step(quantity_unadjusted, step_size)
        
        if quantity <= 0:
            logging.info(f"调整后的数量为0，跳过 {pair} 的交易。")
            continue

        if trade_value < 0:
            sell_quantity = min(current_holdings[pair]['quantity'], quantity)
            sell_quantity_adjusted = adjust_quantity_by_step(sell_quantity, step_size)
            
            if sell_quantity_adjusted <= 0:
                logging.info(f"调整后的卖出数量为0，跳过 {pair} 的卖出。")
                continue

            logging.info(f"再平衡 [卖出]: {pair}, 卖出价值 ${abs(trade_value):.2f} (原始计算数量: {sell_quantity:.6f}, 调整后: {sell_quantity_adjusted})")
            order_result = client.place_order(coin, "SELL", sell_quantity_adjusted)
            logging.info(f"卖单返回: {order_result}")
            if not order_result or not order_result.get("Success"):
                logging.error(f"卖出 {pair} 失败: {order_result}")

        else:
            logging.info(f"再平衡 [买入]: {pair}, 买入价值 ${trade_value:.2f} (原始计算数量: {quantity_unadjusted:.6f}, 调整后: {quantity})")
            order_result = client.place_order(coin, "BUY", quantity)
            logging.info(f"买单返回: {order_result}")
            if not order_result or not order_result.get("Success"):
                logging.error(f"买入 {pair} 失败: {order_result}")

    logging.info("="*10 + " 高级策略执行任务完成 " + "="*10)

if __name__ == "__main__":
    logging.info("高级交易机器人启动...")
    try:
        collect_daily_price_data()
        run_strategy()
    except Exception as e:
        logging.critical(f"首次运行时发生严重错误: {e}", exc_info=True)

    scheduler = BlockingScheduler(timezone="UTC")
    scheduler.add_job(collect_daily_price_data, 'cron', hour=23, minute=59)
    scheduler.add_job(run_strategy, 'cron', hour=0, minute=15)
    
    logging.info("调度器已配置完毕。机器人进入持续运行模式...")
    print("="*60)
    print("高级交易机器人正在运行。按 Ctrl+C 停止。")
    print(f"日志将记录在 {LOG_FILE} 文件中。")
    print(f"调度任务: [采集] 每日 23:59 UTC, [交易] 每日 00:15 UTC")
    print("="*60)
    
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logging.info("交易机器人已手动停止。")