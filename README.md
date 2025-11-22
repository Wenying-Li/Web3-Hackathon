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

## Repository Files
* **final_v1.py** is the first version of code we deployed.
* **final_v2.py** is the one-time update of code during the competition.
* **price_history_new.csv** provides the previous price data before the competition starts for the bot to make decisions at the first few days.
* **portfolio_state.json** is to help the bot transition from the first version of code to the second version.
* **python_demo.py** is used to test the API connection.
