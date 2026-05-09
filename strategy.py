# Hypothesis: This strategy targets breakouts at key Fibonacci retracement levels (0.618, 1.618) from daily ranges, filtered by volume surge and higher timeframe trend (1d EMA50), to capture strong directional moves in both bull and bear markets while minimizing false signals. Fibonacci levels act as natural support/resistance where breakouts often indicate institutional participation, and volume confirmation ensures momentum. The 1d EMA50 filter avoids counter-trend trades during strong trends. Designed for low trade frequency (target: 20-40 trades/year) to reduce fee drag.

# 4h Fibonacci Breakout with Volume and Trend Filter
name = "4h_Fibonacci_Breakout_Volume_Trend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Fibonacci levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 50:
        ema50_1d[49] = np.mean(close_1d[0:50])
        for i in range(50, len(close_1d)):
            ema50_1d[i] = (close_1d[i] * 2 + ema50_1d[i-1] * 48) / 50
    
    # Align 1d EMA50 to 4h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate daily Fibonacci levels (0.618 and 1.618 extensions)
    fib_upper_1d = np.full_like(close_1d, np.nan)
    fib_lower_1d = np.full_like(close_1d, np.nan)
    
    for i in range(len(df_1d)):
        if not (np.isnan(high_1d[i]) or np.isnan(low_1d[i]) or np.isnan(close_1d[i])):
            daily_range = high_1d[i] - low_1d[i]
            fib_upper_1d[i] = close_1d[i] + 1.618 * daily_range
            fib_lower_1d[i] = close_1d[i] - 0.618 * daily_range
    
    # Align Fibonacci levels to 4h timeframe
    fib_upper_1d_aligned = align_htf_to_ltf(prices, df_1d, fib_upper_1d)
    fib_lower_1d_aligned = align_htf_to_ltf(prices, df_1d, fib_lower_1d)
    
    # Volume filter: current volume vs 20-period average
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid_vol = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid_vol] = volume[valid_vol] / vol_ma[valid_vol]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Need 1d EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(fib_upper_1d_aligned[i]) or 
            np.isnan(fib_lower_1d_aligned[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine market conditions
        trend_up = close[i] > ema50_1d_aligned[i]
        volume_surge = volume_ratio[i] > 2.0
        
        if position == 0:
            # Enter long: Uptrend + price breaks above Fibonacci extension + volume surge
            if trend_up and close[i] > fib_upper_1d_aligned[i] and volume_surge:
                signals[i] = 0.25
                position = 1
            # Enter short: Downtrend + price breaks below Fibonacci retracement + volume surge
            elif not trend_up and close[i] < fib_lower_1d_aligned[i] and volume_surge:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Trend turns down OR price breaks below Fibonacci retracement
            if not trend_up or close[i] < fib_lower_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Trend turns up OR price breaks above Fibonacci extension
            if trend_up or close[i] > fib_upper_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

#!/usr/bin/env python3