#!/usr/bin/env python3
name = "4h_382Pivot_LongOnly_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for 38.2% retracement pivot
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Daily high, low, close
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's range (avoid look-ahead)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    # Daily range and 38.2% Fibonacci level from prior close
    daily_range = prev_high - prev_low
    fib_382 = prev_close - 0.382 * daily_range  # 38.2% retracement level
    
    # Align 38.2% level to 4h timeframe
    fib_382_aligned = align_htf_to_ltf(prices, df_1d, fib_382)
    
    # Daily EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: 20-period average on 4h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(fib_382_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position == 1:
                signals[i] = 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume threshold - avoid low-volume false breakouts
        volume_surge = vol_ratio[i] > 1.5
        
        if position == 0:
            # Long: Price breaks above 38.2% retracement level with volume
            # AND price is above daily EMA50 (bullish bias)
            if (close[i] > fib_382_aligned[i] and 
                volume_surge and 
                close[i] > ema_50_aligned[i]):
                signals[i] = 0.25
                position = 1
        else:
            # Exit: price returns below 38.2% level or trend turns bearish
            if (close[i] < fib_382_aligned[i]) or (close[i] < ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
    
    return signals