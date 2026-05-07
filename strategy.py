#!/usr/bin/env python3
# 1d_Range_Breakout_With_Volume_and_Trend_Filter
# Hypothesis: Buy when price breaks above the 20-day high with volume confirmation and weekly trend alignment; sell when breaks below 20-day low with volume and weekly trend alignment.
# Uses weekly trend filter to avoid counter-trend trades. Weekly trend defined as price above/below 50-week EMA.
# Volume confirmation requires current volume > 1.5x 20-day average volume.
# Position size: 0.25 for long/short, 0.0 for flat.
# Designed to work in both bull and bear markets by filtering trades with weekly trend.

timeframe = "1d"
name = "1d_Range_Breakout_With_Volume_and_Trend_Filter"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 50-week EMA for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 20-day high and low for breakout levels
    # Use pandas rolling with min_periods
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-day average volume for volume confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Start after we have 20-day high/low/volume data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or vol_ma_20[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 20-day high with volume confirmation and weekly uptrend
            if (close[i] > high_20[i] and 
                volume[i] > 1.5 * vol_ma_20[i] and 
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-day low with volume confirmation and weekly downtrend
            elif (close[i] < low_20[i] and 
                  volume[i] > 1.5 * vol_ma_20[i] and 
                  close[i] < ema_50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price falls below 20-day low (mean reversion or trend change)
            if close[i] < low_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price rises above 20-day high (mean reversion or trend change)
            if close[i] > high_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals