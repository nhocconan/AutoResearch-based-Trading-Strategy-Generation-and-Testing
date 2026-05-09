#!/usr/bin/env python3
# Hypothesis: 1d Donchian(20) breakout with 1w EMA10 trend filter and volume confirmation
# Long when price breaks above 20-day high with weekly EMA10 uptrend and volume > 1.5x average
# Short when price breaks below 20-day low with weekly EMA10 downtrend and volume > 1.5x average
# Exit when price crosses below 10-day EMA for longs or above 10-day EMA for shorts
# Uses daily price channels for breakouts, weekly trend filter for direction, volume for confirmation
# Designed to capture major trend moves with low frequency to minimize fee drag
# Target: 40-80 total trades over 4 years (10-20/year) with size 0.25

name = "1d_Donchian20_WeeklyEMA10_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1w EMA10 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    ema10_1w = pd.Series(df_1w['close']).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema10_1w_aligned = align_htf_to_ltf(prices, df_1w, ema10_1w)
    
    # Calculate 10-day EMA for exit
    ema10 = pd.Series(close).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Calculate 20-day Donchian channels
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough data for Donchian calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema10_1w_aligned[i]) or np.isnan(ema10[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above 20-day high, weekly EMA10 uptrend, volume spike
            if (close[i] > high_20[i] and 
                ema10_1w_aligned[i] > ema10_1w_aligned[i-1] and  # EMA rising
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below 20-day low, weekly EMA10 downtrend, volume spike
            elif (close[i] < low_20[i] and 
                  ema10_1w_aligned[i] < ema10_1w_aligned[i-1] and  # EMA falling
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below 10-day EMA
            if close[i] < ema10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above 10-day EMA
            if close[i] > ema10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals