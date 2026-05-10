#!/usr/bin/env python3
"""
4h_Camarilla_Pivot_R1S1_Breakout_1dTrend_Volume
Hypothesis: Use 1d Camarilla pivot levels (R1/S1) as breakout levels on 4h chart, with 1d EMA34 trend filter and volume confirmation. Works in both bull and bear by following higher timeframe trend, with volume filter to avoid false breakouts. Target: 75-200 total trades over 4 years (19-50/year).
"""

name = "4h_Camarilla_Pivot_R1S1_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data for pivot calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:  # Need at least 2 days for pivot calculation
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # Using previous day's values (shifted by 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    # First day has no previous day, so set to NaN
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    range_1d = prev_high - prev_low
    camarilla_r1 = prev_close + range_1d * 1.1 / 12
    camarilla_s1 = prev_close - range_1d * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Daily EMA34 for trend filter
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Get price and volume
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 1.3x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema20 * 1.3
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA34 (34) and volume EMA (20)
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_34_aligned[i]) or 
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: uptrend AND price breaks above R1 with volume
            if close[i] > ema_34_aligned[i] and high[i] > r1_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: downtrend AND price breaks below S1 with volume
            elif close[i] < ema_34_aligned[i] and low[i] < s1_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below S1 OR trend turns bearish
            if low[i] < s1_aligned[i] or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above R1 OR trend turns bullish
            if high[i] > r1_aligned[i] or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals