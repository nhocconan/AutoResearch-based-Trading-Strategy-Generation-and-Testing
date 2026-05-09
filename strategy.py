#!/usr/bin/env python3
# Hypothesis: 1h Fibonacci pullback strategy with 4h EMA21 trend filter and volume spike
# Long when price pulls back to 0.618 Fib level in uptrend (4h EMA21 rising) with volume > 1.5x average
# Short when price pulls back to 0.382 Fib level in downtrend (4h EMA21 falling) with volume > 1.5x average
# Exit when price reaches opposite Fib level or shows reversal signs
# Uses 4h for trend direction, 1h for precise entry timing during pullbacks
# Target: 80-120 total trades over 4 years (20-30/year) with size 0.20

name = "1h_Fibonacci_Pullback_4hEMA21_Volume"
timeframe = "1h"
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
    
    # Calculate 4h EMA21 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 21:
        return np.zeros(n)
    
    ema21_4h = pd.Series(df_4h['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema21_4h)
    
    # Calculate 1h swing high/low for Fibonacci levels (20-period lookback)
    def calculate_fib_levels(high_arr, low_arr, lookback=20):
        fib_levels = np.full_like(high_arr, np.nan)
        for i in range(lookback, len(high_arr)):
            window_high = np.max(high_arr[i-lookback:i])
            window_low = np.min(low_arr[i-lookback:i])
            diff = window_high - window_low
            if diff > 0:
                fib_levels[i] = window_high - 0.618 * diff  # 0.618 retracement for longs
                # Also calculate 0.382 for shorts
        return fib_levels
    
    # Calculate Fibonacci levels
    fib_618 = calculate_fib_levels(high, low)
    fib_382 = np.full_like(high, np.nan)
    for i in range(20, len(high)):
        window_high = np.max(high[i-20:i])
        window_low = np.min(low[i-20:i])
        diff = window_high - window_low
        if diff > 0:
            fib_382[i] = window_low + 0.382 * diff  # 0.382 retracement for shorts
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Need enough data for calculations
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema21_4h_aligned[i]) or np.isnan(fib_618[i]) or np.isnan(fib_382[i]) or
            np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: pullback to 0.618 Fib in uptrend with volume spike
            if (low[i] <= fib_618[i] and  # Price touches or goes below 0.618 level
                ema21_4h_aligned[i] > ema21_4h_aligned[i-1] and  # 4h EMA21 rising
                vol_confirm[i]):
                signals[i] = 0.20
                position = 1
            # Enter short: pullback to 0.382 Fib in downtrend with volume spike
            elif (high[i] >= fib_382[i] and  # Price touches or goes above 0.382 level
                  ema21_4h_aligned[i] < ema21_4h_aligned[i-1] and  # 4h EMA21 falling
                  vol_confirm[i]):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price reaches 0.382 Fib level or shows weakness
            if (high[i] >= fib_382[i]) or (close[i] < close[i-1] and low[i] < fib_618[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price reaches 0.618 Fib level or shows strength
            if (low[i] <= fib_618[i]) or (close[i] > close[i-1] and high[i] > fib_382[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals