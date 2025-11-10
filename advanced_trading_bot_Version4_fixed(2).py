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
from datetime import datetime, timezone
from apscheduler.schedulers.blocking import BlockingScheduler

# --- 核心配置 ---
API_KEY = "AsQb4BENw84ot5mAugsXw88o8f37tEK75LJdAveGZvKlvEhbQRQETJElcPZ9CkAE"
SECRET = "uO4TvgFoPcOHPgYPHs398wcqIpAR34khlpbJoPflHdpJGQVUHKosIXTMli7GNUAh"
BASE_URL = "https://mock-api.roostoo.com"

# --- 数据与日志文件 ---
HISTORY_FILE = "price_history_new.csv"
LOG_FILE = "advanced_trading_bot.log"

# --- 策略参数 (已根据文档全面更新) ---
# 组合管理
CASH_ALLOCATION = 0.9           # 投资90%的资金, 保留10%现金
TOP_N = 5                       # 选择Top 5资产
MAX_POSITION_RATIO = 0.25       # 单一资产最大仓位
MIN_ORDER_VALUE = 300           # 最小订单金额 (USD)

# 筛选器参数
VOLUME_FILTER_PERCENTILE = 0.7    # 成交量筛选: Top 30% (即大于70百分位数)
VOLATILITY_FILTER_PERCENTILE = 0.7# 波动率筛选: Top 30% (即大于70百分位数)
VOLATILITY_WINDOW = 7           # 计算波动率的窗口期: 7天
VOLUME_LOOKBACK_DAYS = 5        # 历史平均成交量的回看窗口: 5天

# 动量与时机指标参数
MOMENTUM_LOOKBACK_DAYS = 3      # 动量筛选: 3日回报率
ENTRY_SMA_WINDOW = 7            # 入场时机: 7日SMA
EXIT_SMA_WINDOW = 10            # 出场时机: 10日SMA
RSI_WINDOW = 7                  # RSI窗口期
RSI_ENTRY_MIN = 50              # 入场RSI > 50
RSI_ENTRY_MAX = 70              # 入场RSI < 70 (避免超买)
RSI_EXIT_OVERBOUGHT = 80        # 超买退出RSI阈值
RSI_EXIT_OVERSOLD = 30          # 超卖退出RSI阈值

# 风险管理参数
STOP_LOSS_RATIO = 0.95          # 5% 止损
TAKE_PROFIT_RATIO = 1.33        # 33% 止盈

# --- 日志配置 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

# ---
# API 客户端 (RoostooV3Client)
# ---
class RoostooV3Client:
    """封装对 Roostoo V3 API 的所有请求"""
    def __init__(self, api_key, secret_key):
        self.api_key = api_key
        self.secret = secret_key
        self.base_url = BASE_URL
        logging.info("Roostoo V3 客户端初始化完成")

    def _get_timestamp_ms(self):
        """返回13位毫秒时间戳字符串"""
        return str(int(time.time() * 1000))

    def _generate_signature(self, params):
        """生成API请求签名"""
        query_string = '&'.join([f"{k}={params[k]}" for k in sorted(params.keys())])
        us = self.secret.encode('utf-8')
        m = hmac.new(us, query_string.encode('utf-8'), hashlib.sha256)
        return m.hexdigest()

    def _signed_request(self, method, endpoint, params=None, data=None):
        """执行签名的 GET 或 POST 请求的统一方法"""
        # --- 错误修复 ---
        req_data = params if method == 'GET' else (data if data is not None else {})
        if req_data is None:
            req_data = {}
        req_data["timestamp"] = self._get_timestamp_ms()
        signature = self._generate_signature(req_data)
        
        headers = {"RST-API-KEY": self.api_key, "MSG-SIGNATURE": signature}
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method == 'GET':
                response = requests.get(url, params=req_data, headers=headers, timeout=15)
            else: # POST
                headers['Content-Type'] = 'application/x-www-form-urlencoded'
                response = requests.post(url, data=req_data, headers=headers, timeout=15)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"API {method} 请求失败 ({url}): {e}")
            if e.response: logging.error(f"响应内容: {e.response.text}")
            return None

    def get_ticker(self, pair=None):
        """获取市场行情 (公开端点)"""
        url = f"{self.base_url}/v3/ticker"
        params = {"timestamp": self._get_timestamp_ms()}
        if pair: params["pair"] = pair
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"获取 Ticker 行情失败: {e}")
            return None

    def get_balance(self):
        return self._signed_request("GET", "/v3/balance")

    def place_order(self, coin, side, quantity, price=None):
        data = {"pair": f"{coin}/USD", "side": side.upper(), "quantity": str(quantity)}
        data['type'] = "LIMIT" if price else "MARKET"
        if price: data['price'] = str(price)
        logging.info(f"准备下单: {data}")
        return self._signed_request("POST", "/v3/place_order", data=data)

    def query_order(self, pending_only="FALSE", pair=None):
        data = {"pending_only": pending_only}
        if pair: data["pair"] = pair
        return self._signed_request("POST", "/v3/query_order", data=data)

# ---
# 数据采集
# ---
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

# ---
# 高级策略分析模块
# ---
class StrategyAnalytics:
    """封装所有基于历史数据的复杂计算逻辑"""
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
        
        logging.debug(f"入场检查 ({pair}): Price={last_price:.2f}, SMA7={sma7:.2f} (>{'T' if price_above_sma else 'F'}), "
                      f"RSI7={rsi7:.2f} ({RSI_ENTRY_MIN}<RSI<{RSI_ENTRY_MAX} -> {'T' if rsi_ok else 'F'})")
        
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
# 主策略逻辑
# ---
def run_strategy():
    logging.info("="*10 + " 开始高级策略执行任务 " + "="*10)
    client = RoostooV3Client(API_KEY, SECRET)
    
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
    
    wallet = balance_data.get("Wallet", {})
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

        quantity = abs(trade_value) / price
        if trade_value < 0:
            sell_quantity = min(current_holdings[pair]['quantity'], quantity)
            logging.info(f"再平衡 [卖出]: {pair}, 卖出价值 ${abs(trade_value):.2f} (数量: {sell_quantity:.6f})")
            client.place_order(coin, "SELL", sell_quantity)
        else:
            logging.info(f"再平衡 [买入]: {pair}, 买入价值 ${trade_value:.2f} (数量: {quantity:.6f})")
            client.place_order(coin, "BUY", quantity)

    logging.info("="*10 + " 高级策略执行任务完成 " + "="*10)

# ---
# 启动与调度
# ---
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