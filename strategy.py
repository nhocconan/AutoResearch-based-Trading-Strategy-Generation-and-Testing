#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1w Williams %R mean reversion with 1d EMA34 trend filter
# - Williams %R(14) on weekly timeframe identifies overbought/oversold conditions
# - Enters long when weekly %R crosses above -80 from below (oversold bounce)
# - Enters short when weekly %R crosses below -20 from above (overbought rejection)
# - Only takes trades when 1d EMA34 confirms trend direction (EMA34 slope)
# - Exits when %R returns to -50 (mean reversion midpoint)
# - Designed to capture mean reversion moves within the weekly trend
# - Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position sizing

name = "12h_1wWilliamsR_1dEMA34_MeanReversion"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for Williams %R
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate Williams %R(14) on weekly data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    wr = -100 * (highest_high - close_1w) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    wr = np.where((highest_high - lowest_low) == 0, -50, wr)
    
    # Align Williams %R to 12h timeframe
    wr_12h = align_htf_to_ltf(prices, df_1w, wr)
    
    # Get daily data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate EMA34 on daily close
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align EMA34 to 12h timeframe
    ema_34_12h = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate EMA34 slope for trend direction (positive = uptrend, negative = downtrend)
    ema_slope = np.diff(ema_34_12h, prepend=ema_34_12h[0])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(wr_12h[i]) or np.isnan(ema_34_12h[i]) or np.isnan(ema_slope[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R crosses above -80 from below AND uptrend (EMA slope > 0)
            if i > 0 and wr_12h[i-1] <= -80 and wr_12h[i] > -80 and ema_slope[i] > 0:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 from above AND downtrend (EMA slope < 0)
            elif i > 0 and wr_12h[i-1] >= -20 and wr_12h[i] < -20 and ema_slope[i] < 0:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R returns to -50 (mean reversion midpoint)
            if wr_12h[i] >= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R returns to -50 (mean reversion midpoint)
            if wr_12h[i] <= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals