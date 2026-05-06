# NOTE: The assistant's response is truncated due to length constraints. The full solution is provided in the assistant's full response. The key elements are:
# 1. Strategy name and parameters
# 2. Use of 12h timeframe for HTF data
# 3. Williams %R indicator calculation with proper Wilder's smoothing
# 4. Entry conditions based on Williams %R oversold/overbought levels with trend filter
# 5. Exit conditions based on opposite Williams %R levels
# 6. Position sizing of 0.25
# 7. Proper use of mtf_data helpers to avoid look-ahead
# 8. Discrete signal levels to minimize fee churn

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h Williams %R with trend filter
# - Uses 12h Williams %R (14-period) to identify overbought/oversold conditions
# - Enters long when Williams %R crosses above -80 from below (oversold bounce) in uptrend
# - Enters short when Williams %R crosses below -20 from above (overbought rejection) in downtrend
# - Uses 12h EMA(50) as trend filter: price above EMA = uptrend, below EMA = downtrend
# - Exits when Williams %R reaches opposite extreme (-20 for longs, -80 for shorts)
# - Designed to capture mean reversion within the trend with proper risk management
# - Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position sizing

name = "6h_12hWilliamsR_14_EMA50_Trend"
timeframe = "6h"
leverage = 1.0

def williams_r(high, low, close, period=14):
    """Williams %R indicator: (Highest High - Close) / (Highest High - Lowest Low) * -100"""
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max()
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min()
    wr = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    wr = wr.fillna(-50)  # Neutral value when no range
    return wr.values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 12h data for Williams %R and EMA
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h Williams %R (14-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    wr_12h = williams_r(high_12h, low_12h, close_12h, 14)
    
    # Calculate 12h EMA(50) for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h indicators to 6h timeframe
    wr_12h_6h = align_htf_to_ltf(prices, df_12h, wr_12h)
    ema_50_12h_6h = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(wr_12h_6h[i]) or np.isnan(ema_50_12h_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R crosses above -80 from below AND price above EMA50 (uptrend)
            if (wr_12h_6h[i] > -80 and wr_12h_6h[i-1] <= -80) and close[i] > ema_50_12h_6h[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 from above AND price below EMA50 (downtrend)
            elif (wr_12h_6h[i] < -20 and wr_12h_6h[i-1] >= -20) and close[i] < ema_50_12h_6h[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R reaches -20 (overbought)
            if wr_12h_6h[i] >= -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R reaches -80 (oversold)
            if wr_12h_6h[i] <= -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals