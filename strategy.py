#!/usr/bin/env python3
name = "1h_Fibonacci_Retracement_Trend_With_Volume"
timeframe = "1h"
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
    
    # Get 4h and 1d data for trend and structure
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 50 or len(df_1d) < 20:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 4h EMA200 for long-term trend
    ema200_4h = pd.Series(close_4h).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema200_4h)
    
    # Daily range for Fibonacci levels (using previous day)
    range_1d = high_1d - low_1d
    fib_0_618 = close_1d + 0.618 * range_1d  # 61.8% retracement level for longs
    fib_0_382 = close_1d - 0.382 * range_1d  # 38.2% retracement level for shorts
    
    fib_0_618_aligned = align_htf_to_ltf(prices, df_1d, fib_0_618)
    fib_0_382_aligned = align_htf_to_ltf(prices, df_1d, fib_0_382)
    
    # Volume filter: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 200  # Wait for EMA200
    
    for i in range(start_idx, n):
        if np.isnan(ema200_4h_aligned[i]) or np.isnan(fib_0_618_aligned[i]) or np.isnan(fib_0_382_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if not session_filter[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price pulls back to 61.8% Fib level in uptrend, with volume
            if (close[i] <= fib_0_618_aligned[i] * 1.005 and  # Allow small buffer
                close[i] >= fib_0_618_aligned[i] * 0.995 and
                close[i] > ema200_4h_aligned[i] and
                vol_filter[i]):
                signals[i] = 0.20
                position = 1
            # Short: price bounces off 38.2% Fib level in downtrend, with volume
            elif (close[i] >= fib_0_382_aligned[i] * 0.995 and
                  close[i] <= fib_0_382_aligned[i] * 1.005 and
                  close[i] < ema200_4h_aligned[i] and
                  vol_filter[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price breaks below 38.2% Fib or below EMA200
            if close[i] < fib_0_382_aligned[i] * 0.995 or close[i] < ema200_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price breaks above 61.8% Fib or above EMA200
            if close[i] > fib_0_618_aligned[i] * 1.005 or close[i] > ema200_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals