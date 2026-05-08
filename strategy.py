#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h price action trading using 1d Fibonacci retracement levels and volume confirmation.
# Long when price retraces to 61.8% Fibonacci level from recent swing low and closes above it with volume confirmation.
# Short when price retraces to 38.2% Fibonacci level from recent swing high and closes below it with volume confirmation.
# Uses 12h timeframe with 1d Fibonacci levels for higher timeframe context.
# Designed to work in both bull and bear markets by trading mean reversion within the trend.
# Target: 50-150 total trades over 4 years with controlled frequency to avoid fee drag.

name = "12h_Fibonacci_Retracement_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for Fibonacci calculation
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 2:
        return np.zeros(n)
    
    # Calculate 20-period highest high and lowest low for swing points
    highest_high_20 = pd.Series(df_d['high']).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(df_d['low']).rolling(window=20, min_periods=20).min().values
    
    # Fibonacci levels: 38.2% and 61.8% retracement
    range_20 = highest_high_20 - lowest_low_20
    fib_382 = lowest_low_20 + (range_20 * 0.382)  # Resistance in uptrend, support in downtrend
    fib_618 = lowest_low_20 + (range_20 * 0.618)  # Support in uptrend, resistance in downtrend
    
    # Align Fibonacci levels to 12h timeframe
    fib_382_aligned = align_htf_to_ltf(prices, df_d, fib_382)
    fib_618_aligned = align_htf_to_ltf(prices, df_d, fib_618)
    
    # Volume filter: current volume > 1.3x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(fib_382_aligned[i]) or np.isnan(fib_618_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price at 61.8% support level with volume confirmation
            # Allow small tolerance for price action around the level
            long_cond = (close[i] >= fib_618_aligned[i] * 0.999) and (close[i] <= fib_618_aligned[i] * 1.001) and volume_filter[i]
            # Short conditions: price at 38.2% resistance level with volume confirmation
            short_cond = (close[i] <= fib_382_aligned[i] * 1.001) and (close[i] >= fib_382_aligned[i] * 0.999) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price moves above 38.2% level or below 61.8% level
            if close[i] > fib_382_aligned[i] * 1.001 or close[i] < fib_618_aligned[i] * 0.999:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price moves below 61.8% level or above 38.2% level
            if close[i] < fib_618_aligned[i] * 0.999 or close[i] > fib_382_aligned[i] * 1.001:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals