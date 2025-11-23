# Web3-Hackathon
## 📈 Strategy Documentation

This trading bot implements a **Momentum-based strategy with Volatility adjustments**, designed to capture trends in liquid assets while managing risk through strict entry/exit criteria and position sizing.

### 1. Asset Selection & Filtering
The bot filters the available universe of assets to find the best candidates for trading:
*   **Liquidity Filter:** Selects assets in the top **30%** by trading volume (using a 5-day moving average or real-time data) to ensure trade execution.
*   **Volatility Filter:** Selects assets in the top **30%** by volatility (7-day window). We seek assets with enough movement to generate returns.
*   **Ranking:** The filtered list is ranked by **3-Day Momentum** (percentage change). The top **5 assets** are selected as potential targets.

### 2. Signal Generation
Trading decisions are based on trend-following and mean-reversion indicators:
*   **Entry Conditions:**
    *   **Trend Confirmation:** Price must be above **95% of the 7-day SMA** (`Price > 0.95 * SMA7`).
    *   **Momentum Check:** RSI (7-day) must be between **50 and 70**. This ensures the asset is strengthening but not yet overbought.
*   **Exit Conditions:**
    *   **Trend Reversal:** Price closes below the **10-day SMA**.
    *   **Extreme RSI:** RSI crosses above **80** (Overbought) or drops below **30** (Panic/Oversold).

### 3. Portfolio Construction & Sizing
*   **Allocation:** 90% of the portfolio cash is allocated for trading (`CASH_ALLOCATION = 0.9`).
*   **Inverse Volatility Weighting:** Position sizes are calculated based on the inverse of their volatility (`1 / Volatility`). More volatile assets receive a smaller portion of the capital to normalize risk.
*   **Caps:** Single position size is capped at **5%** of total equity (`MAX_POSITION_RATIO`).

### 4. Risk Management
The bot employs an active risk management system that runs every hour:
*   **Stop-Loss:** Hard stop triggered if the position drops **8%** below the average buy price.
*   **Staged Take-Profit:**
    1.  **+5% Profit:** Sell 30% of the position.
    2.  **+10% Profit:** Sell 40% of the position.
    3.  **+15% Profit:** Sell the remaining 30%.

### 5. Execution Schedule
*   **Data Collection:** Updates price/volume history every 4 hours.
*   **Strategy Rebalancing:** Runs every 4 hours (15 minutes past the hour).
*   **Risk Check (SL/TP):** Runs hourly (30 minutes past the hour).

## 🛠️ Technical Implementation

### 1. Data Pipeline & Persistence (`collect_daily_price_data`)
The bot maintains a local database to record daily prices and volumes to help with looking-back decisions.
*   **Hybrid Data Source:** It combines historical data from `price_history_new.csv` with real-time snapshots from `client.get_ticker()`.
*   **Idempotent Updates:** The function generates a primary key using the current UTC date (`today_utc_str`).
    *   If the date exists in the CSV: It **updates** the row (handling intra-day data changes).
    *   If the date is new: It **appends** a new row.
*   **Benefit:** This ensures the bot always has the latest daily candle to calculate moving averages immediately, without waiting for a daily close.

### 2. Dynamic Filtering Logic (`filter_and_rank_assets`)

*   **Step 1: Dynamic Thresholds**
    The code calculates the 70th percentile (`0.7`) for both volume and volatility across the *entire market*.
    ```python
    volume_threshold = volumes.quantile(0.7)
    volatility_threshold = volatility.quantile(0.7)
    ```
*   **Step 2: Set Intersection**
    It identifies assets that meet *both* criteria using Python set operations:
    ```python
    final_candidates = list(volume_candidates & volatility_candidates)
    ```
*   **Step 3: Momentum Ranking**
    Candidates are sorted by 3-day returns (`pct_change(3)`), prioritizing assets with the strongest recent trend.

### 3. Indicator Implementation (`StrategyAnalytics`)
The code utilizes `pandas_ta` for vectorized technical analysis, ensuring speed even with large datasets.

*   **Entry Signal Logic:**
    Calculates indicators on the full price series but only evaluates the *last* data point (`iloc[-1]`):
    ```python
    sma7 = ta.sma(series, length=7).iloc[-1]
    rsi7 = ta.rsi(series, length=7).iloc[-1]
    # Condition: Price dip buying in an uptrend
    price_above_sma = last_price > (sma7 * 0.95)
    ```
*   **Exit Signal Logic:**
    Includes a "Panic/Euphoria" check. It exits if RSI > 80 (Overbought) OR RSI < 30 (Oversold), protecting against extreme reversals.

### 4. Risk-Parity Position Sizing
The bot does not allocate capital equally. It mathematically smooths risk using **Inverse Volatility Weighting**.

1.  **Calculate Inverse Volatility:** `1 / volatility` for each target.
2.  **Normalize Weights:**
    ```python
    weight = inv_vol / sum_inverse_vol
    weight = min(weight, 0.05)  # Hard cap at 5% allocation
    ```
3.  **Result:** An asset that is twice as volatile as another will receive roughly half the capital allocation.

### 5. Stateful Order Management (`portfolio_state.json`)
To handle **Staged Take-Profits (TP)**, the bot must "remember" which levels have already been sold.

*   **The Problem:** If the price hits +15%, the bot sells 30%. If the price drops to +14% and rises to +15% again, a stateless bot would sell *another* 30%, draining the position.
*   **The Solution (`tp_stage`):**
    The `portfolio_state.json` file tracks a `tp_stage` integer (0-3):
    *   **Stage 0:** Fresh position.
    *   **Stage 1:** Sold at +5%.
    *   **Stage 2:** Sold at +10%.
    *   **Stage 3:** Sold at +15%.
    The code explicitly checks `if profit > level AND tp_stage < level_stage` before executing a sell.

### 6. Automated Scheduling (`APScheduler`) 🕙
The system runs on three separate timelines using `BlockingScheduler`:
1.  **Data Collection (Cron: 00:00, 04:00, ...):** Runs every four hours to capture the prices.
2.  **Strategy Execution (Cron: 00:15, 04:30, ...):** Runs 15 minutes *after* data collection to ensure the dataset is fresh and write operations are complete.
3.  **Risk Daemon (Cron: 00:30, 01:30, ...):** A separate risk management process that runs every hour at minute 30. It purely checks for Stop-Loss and Take-Profit triggers, independent of the trend strategy.

## Repository Files
* **final_v1.py** is the first version of code we deployed.
* **final_v2.py** is the one-time update of code during the competition.
* **price_history_new.csv** provides the previous price data before the competition starts for the bot to make decisions at the first few days.
* **portfolio_state.json** is to help the bot transition from the first version of code to the second version.
* **python_demo.py** is used to test the API connection.
