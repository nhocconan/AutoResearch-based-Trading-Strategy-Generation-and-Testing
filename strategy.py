#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_VolumeSpike
Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume spike (>2.0x median).
Enters long when price breaks above R1 with volume confirmation and bullish 1d trend (price > EMA34).
Enters short when price breaks below S1 with volume confirmation and bearish 1d trend (price < EMA34).
Exits on opposite breakout (price returns to the other level).
Uses discrete position sizing (0.25) to minimize churn. Target: 75-200 trades over 4 years.
Works in both bull and bear markets by following 1d trend filter.
"""

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
    
    # Calculate Camarilla levels for 4h (based on previous 4h bar)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    h_4h = df_4h['high'].values
    l_4h = df_4h['low'].values
    c_4h = df_4h['close'].values
    
    # Previous bar's values for level calculation (to avoid look-ahead)
    h_4h_prev = np.roll(h_4h, 1)
    l_4h_prev = np.roll(l_4h, 1)
    c_4h_prev = np.roll(c_4h, 1)
    h_4h_prev[0] = np.nan
    l_4h_prev[0] = np.nan
    c_4h_prev[0] = np.nan
    
    # Calculate Camarilla R1 and S1 levels
    rng_4h = h_4h_prev - l_4h_prev
    r1_4h = c_4h_prev + (rng_4h * 1.1 / 12)
    s1_4h = c_4h_prev - (rng_4h * 1.1 / 12)
    
    # Align to 4h primary timeframe
    r1_4h_aligned = align_htf_to_ltf(prices, df_4h, r1_4h)
    s1_4h_aligned = align_htf_to_ltf(prices, df_4h, s1_4h)
    
    # Volume confirmation: volume > 2.0x 50-period median (stricter to reduce trades)
    volume_series = pd.Series(volume)
    vol_median = volume_series.rolling(window=50, min_periods=50).median().values
    volume_confirm = volume > (2.0 * vol_median)
    
    # Load 1d data for HTF trend filter (EMA34)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 50-period volume median, 34-period EMA)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_4h_aligned[i]) or np.isnan(s1_4h_aligned[i]) or 
            np.isnan(vol_median[i]) or np.isnan(ema_34_1d_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: price breaks above R1 + volume confirmation + bullish 1d trend
        if close[i] > r1_4h_aligned[i] and volume_confirm[i] and close[i] > ema_34_1d_aligned[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: price breaks below S1 + volume confirmation + bearish 1d trend
        elif close[i] < s1_4h_aligned[i] and volume_confirm[i] and close[i] < ema_34_1d_aligned[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit: opposite breakout (price returns to the other level)
        elif position == 1 and close[i] < s1_4h_aligned[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > r1_4h_aligned[i]:
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

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0