#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Supertrend for trend direction and 1d Chandelier Exit for stop.
# Long when price crosses above Chandelier Exit long level with Supertrend uptrend.
# Short when price crosses below Chandelier Exit short level with Supertrend downtrend.
# Exit when price crosses back below/above Chandelier Exit short/long level.
# Chandelier Exit uses ATR to set dynamic stops, reducing whipsaw in volatile markets.
# Supertrend filters trend direction to avoid counter-trend trades.
# Target: 20-25 trades/year per symbol (80-100 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 12h data ONCE for Supertrend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate True Range and ATR for Supertrend (ATR=10)
    tr1 = np.abs(high_12h[1:] - low_12h[1:])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    atr_period = 10
    atr_12h = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Supertrend calculation
    hl2 = (high_12h + low_12h) / 2
    upper_band = hl2 + (3.0 * atr_12h)
    lower_band = hl2 - (3.0 * atr_12h)
    
    # Initialize Supertrend
    supertrend = np.full_like(close_12h, np.nan)
    dir_12h = np.full_like(close_12h, np.nan)  # 1 for uptrend, -1 for downtrend
    
    # First valid value
    start_idx = atr_period
    if start_idx < len(close_12h):
        supertrend[start_idx] = upper_band[start_idx]
        dir_12h[start_idx] = 1  # start in uptrend
    
    for i in range(start_idx + 1, len(close_12h)):
        if supertrend[i-1] == upper_band[i-1]:
            if close_12h[i] <= upper_band[i]:
                supertrend[i] = upper_band[i]
                dir_12h[i] = 1
            else:
                supertrend[i] = lower_band[i]
                dir_12h[i] = -1
        else:
            if close_12h[i] >= lower_band[i]:
                supertrend[i] = lower_band[i]
                dir_12h[i] = -1
            else:
                supertrend[i] = upper_band[i]
                dir_12h[i] = 1
    
    # Load 1d data ONCE for Chandelier Exit
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range and ATR for Chandelier Exit (ATR=22, multiplier=3.0)
    tr1_d = np.abs(high_1d[1:] - low_1d[1:])
    tr2_d = np.abs(high_1d[1:] - close_1d[:-1])
    tr3_d = np.abs(low_1d[1:] - close_1d[:-1])
    tr_d = np.maximum(tr1_d, np.maximum(tr2_d, tr3_d))
    tr_d = np.concatenate([[np.nan], tr_d])
    
    atr_period_d = 22
    atr_1d = pd.Series(tr_d).ewm(span=atr_period_d, adjust=False, min_periods=atr_period_d).mean().values
    
    # Chandelier Exit calculation
    # Long exit: highest high - ATR * multiplier
    # Short exit: lowest low + ATR * multiplier
    highest_high = np.maximum.accumulate(high_1d)
    lowest_low = np.minimum.accumulate(low_1d)
    
    chandelier_long = highest_high - (3.0 * atr_1d)
    chandelier_short = lowest_low + (3.0 * atr_1d)
    
    # Align indicators to lower timeframe
    dir_12h_aligned = align_htf_to_ltf(prices, df_12h, dir_12h)
    chandelier_long_aligned = align_htf_to_ltf(prices, df_1d, chandelier_long)
    chandelier_short_aligned = align_htf_to_ltf(prices, df_1d, chandelier_short)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(20, 22)  # Need Supertrend and Chandelier Exit
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(dir_12h_aligned[i]) or 
            np.isnan(chandelier_long_aligned[i]) or
            np.isnan(chandelier_short_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Look for Chandelier breakouts
            # Long: price crosses above Chandelier long level AND Supertrend uptrend
            if (close[i] > chandelier_long_aligned[i] and 
                dir_12h_aligned[i] > 0):
                position = 1
                signals[i] = position_size
            # Short: price crosses below Chandelier short level AND Supertrend downtrend
            elif (close[i] < chandelier_short_aligned[i] and 
                  dir_12h_aligned[i] < 0):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below Chandelier short level
            if close[i] < chandelier_short_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above Chandelier long level
            if close[i] > chandelier_long_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Supertrend_Chandelier_Exit_v1"
timeframe = "4h"
leverage = 1.0