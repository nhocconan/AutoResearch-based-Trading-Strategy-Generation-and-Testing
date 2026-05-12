#!/usr/bin/env python3
# 6h_Fibonacci_Retracement_1wTrend_VolumeFilter
# Hypothesis: Combine weekly trend direction with 60-minute Fibonacci retracement levels and volume confirmation.
# In uptrends (price > weekly EMA50), long at 61.8% retracement of weekly swing low-high.
# In downtrends (price < weekly EMA50), short at 38.2% retracement of weekly swing high-low.
# Uses volume spike (>1.5x 20-period MA) for confirmation. Designed to catch pullbacks in strong trends.
# Weekly trend filter reduces whipsaws in sideways markets. Targets 15-30 trades/year to minimize fee drag.

name = "6h_Fibonacci_Retracement_1wTrend_VolumeFilter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly trend filter: EMA50 on weekly closes
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    weekly_close = df_1w['close'].values
    weekly_ema50 = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_ema50_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema50)
    
    # Weekly swing high-low for Fibonacci levels (using 20-period lookback)
    def rolling_max(arr, window):
        return pd.Series(arr).rolling(window=window, min_periods=window).max().values
    
    def rolling_min(arr, window):
        return pd.Series(arr).rolling(window=window, min_periods=window).min().values
    
    weekly_high = rolling_max(weekly_close, 20)
    weekly_low = rolling_min(weekly_close, 20)
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_low)
    
    # Calculate Fibonacci retracement levels
    weekly_range = weekly_high_aligned - weekly_low_aligned
    fib_382 = weekly_low_aligned + 0.382 * weekly_range  # 38.2% retracement
    fib_618 = weekly_low_aligned + 0.618 * weekly_range  # 61.8% retracement
    
    # Volume confirmation: 20-period moving average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(weekly_ema50_aligned[i]) or np.isnan(fib_382[i]) or 
            np.isnan(fib_618[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Uptrend (price > weekly EMA50) + price at 61.8% retracement + volume spike
            if (close[i] > weekly_ema50_aligned[i] and 
                abs(close[i] - fib_618[i]) < 0.005 * close[i] and  # within 0.5% of 61.8% level
                volume[i] > vol_ma[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Downtrend (price < weekly EMA50) + price at 38.2% retracement + volume spike
            elif (close[i] < weekly_ema50_aligned[i] and 
                  abs(close[i] - fib_382[i]) < 0.005 * close[i] and  # within 0.5% of 38.2% level
                  volume[i] > vol_ma[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below 38.2% retracement or weekly trend turns down
            if close[i] < fib_382[i] or close[i] < weekly_ema50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above 61.8% retracement or weekly trend turns up
            if close[i] > fib_618[i] or close[i] > weekly_ema50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals