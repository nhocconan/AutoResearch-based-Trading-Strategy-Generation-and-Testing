#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_v1
Hypothesis: Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume spike confirmation.
Goes long when price breaks above R1 with bullish 1d trend and volume spike; short when price breaks below S1 with bearish 1d trend and volume spike.
Uses discrete position sizing (0.0, ±0.25) to minimize fee churn. Designed for 75-150 total trades over 4 years (19-38/year).
Works in both bull and bear markets by combining price structure (Camarilla pivots) with trend (1d EMA) and volume filters.
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
    
    # Calculate Camarilla pivot levels (R1, S1) from previous day
    # Camarilla: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    # Using 1d data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Camarilla R1 and S1 for each day
    camarilla_r1_1d = close_1d + 1.1 * (high_1d - low_1d) / 12
    camarilla_s1_1d = close_1d - 1.1 * (high_1d - low_1d) / 12
    
    # Align to 4h timeframe (these levels are valid for the entire day)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1_1d)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1_1d)
    
    # 1d EMA34 trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 1.5 * 20-period EMA volume
    avg_volume = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for EMA, 20 for volume EMA)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Discrete position sizing
        base_size = 0.25
        
        # Long logic: price breaks above R1 + bullish 1d trend (price > EMA34) + volume spike
        if close[i] > camarilla_r1_aligned[i] and close[i] > ema_34_1d_aligned[i] and volume_spike[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: price breaks below S1 + bearish 1d trend (price < EMA34) + volume spike
        elif close[i] < camarilla_s1_aligned[i] and close[i] < ema_34_1d_aligned[i] and volume_spike[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit conditions: price returns to midline or loss of volume confirmation
        elif position == 1 and (close[i] < camarilla_r1_aligned[i] or not volume_spike[i]):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close[i] > camarilla_s1_aligned[i] or not volume_spike[i]):
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

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0