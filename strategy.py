#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_v2
Hypothesis: 12h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume spike (>2.5x 20-period median).
Enters long when price closes above R1 with volume confirmation and bullish 1d trend.
Enters short when price closes below S1 with volume confirmation and bearish 1d trend.
Exits on opposite Camarilla level touch (long exits at S1, short exits at R1).
Uses discrete position sizing (0.25) and stricter volume filter to reduce trades and avoid overtrading.
Designed for 12h timeframe targeting 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume confirmation: volume > 2.5x 20-period median (stricter to reduce trades)
    volume_series = pd.Series(volume)
    vol_median = volume_series.rolling(window=20, min_periods=20).median().values
    volume_confirm = volume > (2.5 * vol_median)
    
    # Load 1d data for HTF trend filter (EMA34) and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Camarilla levels for 1d (R1, S1)
    # Camarilla: R1 = close + 1.1*(high-low)/4, S1 = close - 1.1*(high-low)/4
    camarilla_r1 = close_1d + 1.1 * (high_1d - low_1d) / 4
    camarilla_s1 = close_1d - 1.1 * (high_1d - low_1d) / 4
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 20-period volume median, 34-period EMA)
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(vol_median[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: close above R1 + volume confirm + bullish 1d trend
        if close[i] > camarilla_r1_aligned[i] and volume_confirm[i] and close[i] > ema34_1d_aligned[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: close below S1 + volume confirm + bearish 1d trend
        elif close[i] < camarilla_s1_aligned[i] and volume_confirm[i] and close[i] < ema34_1d_aligned[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit: long exits when price touches S1, short exits when price touches R1
        elif position == 1 and close[i] <= camarilla_s1_aligned[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] >= camarilla_r1_aligned[i]:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_v2"
timeframe = "12h"
leverage = 1.0