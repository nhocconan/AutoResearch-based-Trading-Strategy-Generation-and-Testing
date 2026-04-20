#!/usr/bin/env python3
"""
1d_Camarilla_Pivot_R1S1_Breakout_Volume_Trend_Filter
Hypothesis: Trade Camarilla pivot level breakouts on daily with volume confirmation and weekly trend filter.
Long when price breaks above R1 with volume spike and weekly uptrend; short when breaks below S1 with volume spike and weekly downtrend.
Uses volume > 1.8x 20-day average for breakout confirmation to reduce trade frequency and avoid false signals.
Target: 20-60 total trades over 4 years (5-15/year) with position size 0.25.
Works in bull/bear: weekly trend filter avoids counter-trend trades, volume threshold filters weak breakouts.
"""

name = "1d_Camarilla_Pivot_R1S1_Breakout_Volume_Trend_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA50 for trend filter
    def ema(values, period):
        result = np.full_like(values, np.nan)
        if len(values) >= period:
            multiplier = 2.0 / (period + 1)
            result[period-1] = np.mean(values[:period])
            for i in range(period, len(values)):
                result[i] = multiplier * values[i] + (1 - multiplier) * result[i-1]
        return result
    
    ema50_1w = ema(close_1w, 50)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate Camarilla pivot levels for previous day (using previous day's OHLC)
    # We need to shift the OHLC data by 1 to get previous day's values
    high_prev = np.roll(high, 1)
    low_prev = np.roll(low, 1)
    close_prev = np.roll(close, 1)
    high_prev[0] = np.nan  # First day has no previous day
    low_prev[0] = np.nan
    close_prev[0] = np.nan
    
    # Calculate pivot point
    pivot = (high_prev + low_prev + close_prev) / 3.0
    
    # Calculate R1 and S1 levels
    r1 = close_prev + (1.1 / 12.0) * (high_prev - low_prev)
    s1 = close_prev - (1.1 / 12.0) * (high_prev - low_prev)
    
    # Calculate volume spike (volume > 1.8x 20-day average)
    vol_ma20 = np.full_like(volume, np.nan)
    for i in range(20, len(volume)):
        vol_ma20[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.8 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 21  # Need at least 20 days for volume MA and 1 for previous day data
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(r1[i]) or np.isnan(s1[i]) or 
            np.isnan(close[i]) or np.isnan(volume[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 with volume spike AND weekly uptrend (close > EMA50)
            if close[i] > r1[i] and volume_spike[i] and close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume spike AND weekly downtrend (close < EMA50)
            elif close[i] < s1[i] and volume_spike[i] and close[i] < ema50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below S1 OR weekly trend turns down
            if close[i] < s1[i] or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above R1 OR weekly trend turns up
            if close[i] > r1[i] or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals