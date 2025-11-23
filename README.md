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

The bot is built in **Python 3** and leverages a modular architecture to separate API interaction, data analysis, and execution logic.

### 1. Core Libraries & Stack
*   **Pandas:** Used for all data manipulation. The bot maintains a local CSV database (`price_history_new.csv`) and uses Pandas DataFrames for vectorized calculations of momentum, volatility, and moving averages.
*   **Pandas-TA:** Dedicated library for calculating technical indicators (SMA, RSI) efficiently.
*   **APScheduler:** A `BlockingScheduler` is used to orchestrate tasks using Cron-style triggers, ensuring regular timing for market scans (4-hour intervals) and risk checks (hourly).
*   **Requests & HMAC:** Use `RoostooV3Client` class to handle the API calls.

### 2. Data & State Management
The bot is designed to be stateless regarding the runtime memory but stateful via local files to survive restarts:
*   **Historical Data (`price_history_new.csv`):** To avoid API rate limits and ensure sufficient lookback for indicators (SMA, Volatility), the bot incrementally updates this CSV with daily OHLCV data.
*   **Portfolio State (`portfolio_state.json`):** The bot records the entry price and the stage of risk management.
    *   **Tracks:** `avg_buy_price` (for accurate Stop-Loss calculations), `original_quantity`, and `tp_stage` (0, 1, 2, or 3) to ensure the risk management process is handled correctly.
### 3. Quantitative Logic
*   **Volatility Calculation:** Standard deviation of percentage returns over a 7-day rolling window.
*   **Inverse Volatility Weighting:**
    The code implements a risk-parity approach where position size is inversely proportional to risk.
    ```python
    inverse_volatilities = {t['pair']: 1 / t['volatility'] ...}
    weight = inv_vol / sum_inverse_vol
    ```
    This ensures that highly volatile assets receive a smaller allocation, smoothing out the portfolio's equity curve.

### 4. Execution Logic
*   **Rebalancing:** The `run_strategy` function calculates the difference between the *Target Value* (Equity * Weight) and *Current Value*.
    *   If `Target > Current`: **Buy** to fill the gap.
    *   If `Target < Current`: **Sell** to trim the position.
*   **Filters:** Uses Pandas `quantile(0.7)` to dynamically determine the top 30% thresholds for volume and volatility, ensuring the bot adapts to changing market conditions rather than using hardcoded values.

## Repository Files
* **final_v1.py** is the first version of code we deployed.
* **final_v2.py** is the one-time update of code during the competition.
* **price_history_new.csv** provides the previous price data before the competition starts for the bot to make decisions at the first few days.
* **portfolio_state.json** is to help the bot transition from the first version of code to the second version.
* **python_demo.py** is used to test the API connection.
