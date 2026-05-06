#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using daily Williams %R extremes with 12h EMA trend filter
# - Uses daily Williams %R (14-period) for overbought/oversold signals
# - Uses 12h EMA (50-period) for trend direction filter
# - Enters long when Williams %R < -80 (oversold) and price > 12h EMA50
# - Enters short when Williams %R > -20 (overbought) and price < 12h EMA50
# - Exits when Williams %R returns to neutral range (-50) or opposite extreme
# - Designed to capture mean reversion in extreme conditions with trend alignment
# - Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position sizing

name = "6h_1dWilliamsR_14_12hEMA50_Trend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for Williams %R
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d Williams %R (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Williams %R = -100 * (HH - Close) / (HH - LL)
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # Avoid division by zero
    
    # Align 1d Williams %R to 6h timeframe
    williams_r_6h = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_6h = align_htf_to_ltf(prices, df_12h, ema_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(williams_r_6h[i]) or np.isnan(ema_50_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: oversold (Williams %R < -80) and price above 12h EMA50 (uptrend)
            if williams_r_6h[i] < -80 and close[i] > ema_50_6h[i]:
                signals[i] = 0.25
                position = 1
            # Short: overbought (Williams %R > -20) and price below 12h EMA50 (downtrend)
            elif williams_r_6h[i] > -20 and close[i] < ema_50_6h[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R returns to neutral (-50) or overbought (> -20)
            if williams_r_6h[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R returns to neutral (-50) or oversold (< -80)
            if williams_r_6h[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals