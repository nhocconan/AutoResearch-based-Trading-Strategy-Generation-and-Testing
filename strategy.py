#!/usr/bin/env python3
# 12h_1d_1w_TripleTimeframe_Camarilla_R1_S1_Breakout_Volume
# Hypothesis: Triple timeframe alignment (12h, 1d, 1w) using Camarilla R1/S1 breakouts with volume confirmation.
# Long when price > daily EMA34 and weekly EMA34 (bullish trend) and breaks above daily R1 with volume spike.
# Short when price < daily EMA34 and weekly EMA34 (bearish trend) and breaks below daily S1 with volume spike.
# Target: 20-40 trades/year, using discrete position sizing to minimize fee churn.

name = "12h_1d_1w_TripleTimeframe_Camarilla_R1_S1_Breakout_Volume"
timeframe = "12h"
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
    
    # Get daily and weekly data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 35 or len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate daily pivot points
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = pivot_1d + (high_1d - low_1d) * 1.1 / 12
    s1_1d = pivot_1d - (high_1d - low_1d) * 1.1 / 12
    
    # Calculate daily EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate weekly EMA34 for trend filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate volume spike detector (volume > 1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=1).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    # Align all indicators to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(ema34_1w_aligned[i]) or
            np.isnan(volume_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: bullish trend on both timeframes + R1 breakout + volume spike
            if (close[i] > ema34_1d_aligned[i] and close[i] > ema34_1w_aligned[i] and
                close[i] > r1_aligned[i] and volume_spike_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: bearish trend on both timeframes + S1 breakout + volume spike
            elif (close[i] < ema34_1d_aligned[i] and close[i] < ema34_1w_aligned[i] and
                  close[i] < s1_aligned[i] and volume_spike_aligned[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if trend breaks or price breaks below S1
            if (close[i] < ema34_1d_aligned[i] or close[i] < ema34_1w_aligned[i] or
                close[i] < s1_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if trend breaks or price breaks above R1
            if (close[i] > ema34_1d_aligned[i] or close[i] > ema34_1w_aligned[i] or
                close[i] > r1_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals