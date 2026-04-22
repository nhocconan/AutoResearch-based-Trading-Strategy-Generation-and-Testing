#!/usr/bin/env python3
"""
Hypothesis: 1h Fibonacci Retracement Breakout with 4h Trend and 1d Volume Filter.
- Long when: price breaks above 4h 61.8% Fibonacci retracement level AND 1d volume > 20-period average volume AND price > 4h EMA50 (trend filter)
- Short when: price breaks below 4h 38.2% Fibonacci retracement level AND 1d volume > 20-period average volume AND price < 4h EMA50
- Exit when price crosses the 4h 50% Fibonacci level
- Uses Fibonacci levels from prior 4h swing high/low to identify institutional entry zones
- Volume filter ensures participation, trend filter avoids counter-trend trades
- Designed for 60-150 total trades over 4 years (15-37/year) to minimize fee drag
- Works in bull/bear markets by following 4h trend with institutional volume confirmation
"""

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
    
    # Load 4h data for Fibonacci levels and trend filter - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Fibonacci levels from 4h swing high/low (20-period lookback)
    # Find swing high and low over 20 periods
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    roll_max = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    roll_min = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Calculate Fibonacci levels: 0%, 38.2%, 50%, 61.8%, 100%
    fib_range = roll_max - roll_min
    fib_382 = roll_max - 0.382 * fib_range  # 38.2% level
    fib_50 = roll_max - 0.500 * fib_range   # 50% level
    fib_618 = roll_max - 0.618 * fib_range  # 61.8% level
    
    # Align Fibonacci levels to 1h timeframe
    fib_382_aligned = align_htf_to_ltf(prices, df_4h, fib_382)
    fib_50_aligned = align_htf_to_ltf(prices, df_4h, fib_50)
    fib_618_aligned = align_htf_to_ltf(prices, df_4h, fib_618)
    
    # Load 1d data for volume filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    avg_vol_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    avg_vol_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(fib_382_aligned[i]) or 
            np.isnan(fib_50_aligned[i]) or np.isnan(fib_618_aligned[i]) or 
            np.isnan(avg_vol_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above 61.8% Fib level, volume confirmation, uptrend
            if (close[i] > fib_618_aligned[i] and 
                volume > avg_vol_1d_aligned[i] and 
                close[i] > ema_50_4h_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short: Price breaks below 38.2% Fib level, volume confirmation, downtrend
            elif (close[i] < fib_382_aligned[i] and 
                  volume > avg_vol_1d_aligned[i] and 
                  close[i] < ema_50_4h_aligned[i]):
                signals[i] = -0.20
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price crosses below 50% Fib level
                if close[i] < fib_50_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price crosses above 50% Fib level
                if close[i] > fib_50_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_Fibonacci_Breakout_4hTrend_1dVolume"
timeframe = "1h"
leverage = 1.0