#!/usr/bin/env python3
"""
1H_Camarilla_R1_S1_Breakout_4hTrend_Volume
Hypothesis: 1h price breaks above/below 4h Camarilla R1/S1 levels with 4h EMA34 trend confirmation and volume spike.
Uses 4h for signal direction (trend + levels) and 1h for entry timing. Targets 15-35 trades/year to avoid fee drag.
Works in bull/bear markets: R1/S1 breakouts capture meaningful moves, EMA34 filter avoids counter-trend trades.
Volume confirmation ensures breakout legitimacy.
"""
name = "1H_Camarilla_R1_S1_Breakout_4hTrend_Volume"
timeframe = "1h"
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
    
    # Get 4h data for Camarilla levels, EMA trend, and volume average
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    # Calculate 4h Camarilla levels (R1, S1)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    pivot = (high_4h + low_4h + close_4h) / 3
    range_4h = high_4h - low_4h
    r1 = pivot + (range_4h * 1.1 / 4)  # R1 level
    s1 = pivot - (range_4h * 1.1 / 4)  # S1 level
    r1_aligned = align_htf_to_ltf(prices, df_4h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_4h, s1)
    
    # Calculate 4h EMA34 for trend direction
    close_4h_series = pd.Series(df_4h['close'])
    ema_34 = close_4h_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_4h, ema_34)
    
    # Volume filter: current 1h volume > 2.0 x 20-period average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_exit = 0  # bars since last exit to prevent overtrading
    
    start_idx = max(34, 20)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        bars_since_exit += 1
        
        # Skip if any data is not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            continue
        
        if position == 0:
            # Minimum 4 bars between trades (4 hours on 1h TF) to reduce frequency
            if bars_since_exit < 4:
                continue
                
            # Long: price breaks above R1 with EMA34 uptrend and volume spike
            if (close[i] > r1_aligned[i] and close[i-1] <= r1_aligned[i-1] and 
                close[i] > ema_34_aligned[i] and volume_filter[i]):
                signals[i] = 0.20
                position = 1
                bars_since_exit = 0
            # Short: price breaks below S1 with EMA34 downtrend and volume spike
            elif (close[i] < s1_aligned[i] and close[i-1] >= s1_aligned[i-1] and 
                  close[i] < ema_34_aligned[i] and volume_filter[i]):
                signals[i] = -0.20
                position = -1
                bars_since_exit = 0
        elif position != 0:
            # Exit: price returns to opposite EMA34 side (trend reversal)
            if position == 1 and close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            elif position == -1 and close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                # Hold position
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals