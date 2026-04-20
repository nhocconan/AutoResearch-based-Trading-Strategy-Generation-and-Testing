#!/usr/bin/env python3
"""
1d_Donchian_Breakout_WeeklyTrend_Filter
Hypothesis: Trade daily Donchian(20) breakouts with weekly trend filter (EMA50) and volume confirmation.
Long when price breaks above 20-day high with volume spike and weekly uptrend; short when breaks below 20-day low with volume spike and weekly downtrend.
Uses volume spike (volume > 1.5x 20-day average) to confirm breakout strength.
Target: 30-100 total trades over 4 years (7-25/year) with position size 0.25.
Works in bull/bear: weekly trend filter avoids counter-trend trades, volume confirmation reduces false breakouts.
"""

name = "1d_Donchian_Breakout_WeeklyTrend_Filter"
timeframe = "1d"
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
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
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
    
    # Calculate volume spike (volume > 1.5x 20-day average)
    vol_ma20 = np.full_like(volume, np.nan)
    for i in range(20, len(volume)):
        vol_ma20[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma20)
    
    # Calculate Donchian channels (20-period)
    high_max20 = np.full_like(high, np.nan)
    low_min20 = np.full_like(low, np.nan)
    
    for i in range(20, len(high)):
        high_max20[i] = np.max(high[i-20:i])
        low_min20[i] = np.min(low[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(high_max20[i]) or np.isnan(low_min20[i]) or 
            np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above 20-day high with volume spike AND weekly uptrend (close > weekly EMA50)
            if close[i] > high_max20[i] and volume_spike[i] and close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-day low with volume spike AND weekly downtrend (close < weekly EMA50)
            elif close[i] < low_min20[i] and volume_spike[i] and close[i] < ema50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below 20-day low OR weekly trend turns down
            if close[i] < low_min20[i] or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above 20-day high OR weekly trend turns up
            if close[i] > high_max20[i] or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals