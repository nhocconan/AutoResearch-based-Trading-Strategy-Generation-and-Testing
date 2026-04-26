#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike
Hypothesis: 12h Camarilla R1/S1 breakout with 1w EMA50 trend filter and volume spike (>2.0x 20-period median).
Enters long when price closes above R1 with volume confirmation and bullish 1w trend.
Enters short when price closes below S1 with volume confirmation and bearish 1w trend.
Exits on opposite Camarilla level touch (long exits at S1, short exits at R1).
Uses discrete position sizing (0.25) to minimize churn. Target: 75-200 trades over 4 years.
Works in both bull and bear markets by following 1w trend filter and requiring volume confirmation.
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
    
    # Volume confirmation: volume > 2.0x 20-period median
    volume_series = pd.Series(volume)
    vol_median = volume_series.rolling(window=20, min_periods=20).median().values
    volume_confirm = volume > (2.0 * vol_median)
    
    # Load 1w data for HTF trend filter (EMA50) and Camarilla calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # 1w EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate Camarilla levels for 1w (R1, S1)
    # Camarilla: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    camarilla_r1 = close_1w + 1.1 * (high_1w - low_1w) / 12
    camarilla_s1 = close_1w - 1.1 * (high_1w - low_1w) / 12
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 20-period volume median, 50-period EMA)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(vol_median[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: close above R1 + volume confirm + bullish 1w trend
        if close[i] > camarilla_r1_aligned[i] and volume_confirm[i] and close[i] > ema50_1w_aligned[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: close below S1 + volume confirm + bearish 1w trend
        elif close[i] < camarilla_s1_aligned[i] and volume_confirm[i] and close[i] < ema50_1w_aligned[i]:
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

name = "12h_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0