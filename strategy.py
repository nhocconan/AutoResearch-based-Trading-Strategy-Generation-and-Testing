#!/usr/bin/env python3
# Hypothesis: 1d Donchian(20) breakout with 1w EMA20 trend filter and volume confirmation
# Long when price breaks above 20-day high with 1w EMA20 uptrend and volume > 1.5x average
# Short when price breaks below 20-day low with 1w EMA20 downtrend and volume > 1.5x average
# Exit when price retraces to 10-day EMA or opposite 10-day Donchian channel
# Uses daily price structure for breakouts, weekly trend for bias, volume for conviction
# Designed to capture medium-term trends in both bull and bear markets with low frequency
# Target: 30-70 total trades over 4 years (7-17/year) with size 0.25

name = "1d_Donchian20_1wEMA20_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1w EMA20 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    ema20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Calculate 10-day EMA for exit
    ema10 = pd.Series(close).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Calculate 20-day Donchian channels
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Need enough data for Donchian calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema20_1w_aligned[i]) or np.isnan(ema10[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above 20-day high, 1w EMA20 uptrend, volume spike
            if (close[i] > high_20[i] and 
                ema20_1w_aligned[i] > ema20_1w_aligned[i-1] and  # EMA rising
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below 20-day low, 1w EMA20 downtrend, volume spike
            elif (close[i] < low_20[i] and 
                  ema20_1w_aligned[i] < ema20_1w_aligned[i-1] and  # EMA falling
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price retraces to 10-day EMA or breaks below 10-day low
            if (close[i] <= ema10[i]) or (close[i] < pd.Series(low).rolling(window=10, min_periods=10).min().values[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price retraces to 10-day EMA or breaks above 10-day high
            if (close[i] >= ema10[i]) or (close[i] > pd.Series(high).rolling(window=10, min_periods=10).max().values[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals