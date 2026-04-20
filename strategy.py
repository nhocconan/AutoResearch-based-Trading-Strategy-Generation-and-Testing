#!/usr/bin/env python3
# 4h_12h_Camarilla_R1S1_TrendFollow_VolumeFilter
# Hypothesis: Daily Camarilla R1/S1 breakouts on 4h timeframe with 12h EMA trend filter and volume confirmation.
# Uses 12h EMA for trend (more responsive than daily) and volume spike to avoid false breakouts.
# Target: 20-40 trades/year per symbol for balance between signal quality and frequency.

name = "4h_12h_Camarilla_R1S1_TrendFollow_VolumeFilter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 35:
        return np.zeros(n)
    
    # Calculate 12h pivot points
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    pivot_12h = (high_12h + low_12h + close_12h) / 3.0
    r1_12h = pivot_12h + (high_12h - low_12h) * 1.1 / 12
    s1_12h = pivot_12h - (high_12h - low_12h) * 1.1 / 12
    
    # Calculate 12h EMA34 for trend filter
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate volume average for spike detection
    vol_ma_12h = pd.Series(volume).rolling(window=24, min_periods=24).mean().values  # 24*4h = 4 days
    
    # Align 12h indicators to 4h timeframe
    r1_12h_aligned = align_htf_to_ltf(prices, df_12h, r1_12h)
    s1_12h_aligned = align_htf_to_ltf(prices, df_12h, s1_12h)
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_12h_aligned[i]) or np.isnan(s1_12h_aligned[i]) or 
            np.isnan(ema34_12h_aligned[i]) or np.isnan(vol_ma_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike condition: current volume > 1.5 * 12h average volume
        volume_spike = volume[i] > 1.5 * vol_ma_12h_aligned[i]
        
        if position == 0:
            # Long: price > 12h EMA34 (uptrend) and breaks above R1 with volume
            if close[i] > ema34_12h_aligned[i] and close[i] > r1_12h_aligned[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: price < 12h EMA34 (downtrend) and breaks below S1 with volume
            elif close[i] < ema34_12h_aligned[i] and close[i] < s1_12h_aligned[i] and volume_spike:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below S1 (reversal) or trend changes
            if close[i] < s1_12h_aligned[i] or close[i] < ema34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above R1 (reversal) or trend changes
            if close[i] > r1_12h_aligned[i] or close[i] > ema34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals