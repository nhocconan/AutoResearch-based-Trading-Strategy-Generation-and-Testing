#!/usr/bin/env python3
name = "6H_LR_Slope_1dTrend_Confirm"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Linear regression slope on 6h close (36 periods = 9 days)
    window = 36
    slope = np.full(n, np.nan)
    for i in range(window-1, n):
        y = close[i-window+1:i+1]
        x = np.arange(window)
        if np.all(np.isnan(y)) or np.all(np.isnan(x)):
            continue
        # Remove NaNs if any (shouldn't happen with proper data)
        valid = ~np.isnan(y)
        if np.sum(valid) < 2:
            continue
        y_valid = y[valid]
        x_valid = x[valid]
        slope[i] = np.polyfit(x_valid, y_valid, 1)[0]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = max(window, 50)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(slope[i]) or np.isnan(ema50_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend from 1d EMA50
        uptrend = close[i] > ema50_1d_aligned[i]
        downtrend = close[i] < ema50_1d_aligned[i]
        
        if position == 0:
            # Enter long: Uptrend + positive LR slope
            if uptrend and slope[i] > 0:
                signals[i] = 0.25
                position = 1
            # Enter short: Downtrend + negative LR slope
            elif downtrend and slope[i] < 0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Downtrend OR negative LR slope
            if not uptrend or slope[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Uptrend OR positive LR slope
            if not downtrend or slope[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals