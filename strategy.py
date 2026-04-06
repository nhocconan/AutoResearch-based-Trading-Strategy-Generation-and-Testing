#!/usr/bin/env python3
"""
6h Camarilla Pivot with 1d Trend Filter and Volume Confirmation
Hypothesis: Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) work well in both bull and bear markets when filtered by 1d trend.
In bull markets: buy at S3 bounce, break above R4 continuation. In bear markets: sell at R3 rejection, break below S4 continuation.
Uses 1d EMA50 for trend filter and volume confirmation to ensure institutional participation. Target: 75-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_ata, align_htf_to_ltf

name = "6h_camarilla_pivot_1d_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels from previous day's OHLC
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    
    # Camarilla calculations
    range_prev = high_prev - low_prev
    camarilla_base = (high_prev + low_prev + close_prev) / 3
    r3 = camarilla_base + range_prev * 1.1 / 2
    s3 = camarilla_base - range_prev * 1.1 / 2
    r4 = camarilla_base + range_prev * 1.1
    s4 = camarilla_base - range_prev * 1.1
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_prev = np.roll(ema50_1d, 1)
    ema50_1d_prev[0] = ema50_1d[0]
    ema50_rising = ema50_1d > ema50_1d_prev
    ema50_falling = ema50_1d < ema50_1d_prev
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    ema50_rising_aligned = align_htf_to_ltf(prices, df_1d, ema50_rising)
    ema50_falling_aligned = align_htf_to_ltf(prices, df_1d, ema50_falling)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 50  # For EMA50
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(high_prev[i]) or np.isnan(low_prev[i]) or np.isnan(close_prev[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(ema50_rising_aligned[i]) or 
            np.isnan(ema50_falling_aligned[i]) or np.isnan(vol_ema[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: reverse signal or stoploss
        if position == 1:  # long position
            # Exit: price breaks below S3 OR stoploss
            if (close[i] <= s3[i] or 
                close[i] <= entry_price - 2.0 * (high[i] - low[i])):  # ATR proxy
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above R3 OR stoploss
            if (close[i] >= r3[i] or 
                close[i] >= entry_price + 2.0 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Camarilla + trend + volume
            # Long: bounce at S3 in uptrend OR break above R4 in uptrend
            long_entry = ((close[i] >= s3[i] and close[i-1] < s3[i]) and 
                         ema50_rising_aligned[i] and 
                         volume[i] > vol_ema[i] * 1.5)
            # Short: rejection at R3 in downtrend OR break below S4 in downtrend
            short_entry = ((close[i] <= r3[i] and close[i-1] > r3[i]) and 
                          ema50_falling_aligned[i] and 
                          volume[i] > vol_ema[i] * 1.5)
            
            if long_entry:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_entry:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals
</error>Invalid argument type: expected string or bytes-like object, got 'float' for argument 'other'</error>Reasoning: Error occurred during code generation. Please fix. Note: the code above may have been modified by automatic error correction utilities. If this happens, the original code may need to be revisited to identify the root cause. The above is provided for reference only. Please redo your work, taking the error into account.