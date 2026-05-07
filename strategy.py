#!/usr/bin/env python3
"""
1H_Camarilla_R1_S1_Breakout_4hTrend_1dVolume
Hypothesis: 1h price breaks above/below 4h Camarilla R1/S1 levels with 4h EMA34 trend confirmation and 1d volume spike.
Works in bull/bear markets: R1/S1 breakouts capture strong moves while avoiding minor retracements.
EMA34 filter ensures alignment with 4h trend, 1d volume confirmation validates breakout strength.
Uses 4h for signal direction, 1h only for entry timing. Targets 15-37 trades/year on 1h timeframe.
Session filter (08-20 UTC) reduces noise trades.
"""
name = "1H_Camarilla_R1_S1_Breakout_4hTrend_1dVolume"
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
    
    # Precompute session hours (08-20 UTC)
    hours = prices.index.hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for Camarilla levels and EMA trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    # Calculate 4h Camarilla levels (R1, S1)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    pivot_4h = (high_4h + low_4h + close_4h) / 3
    range_4h = high_4h - low_4h
    r1_4h = pivot_4h + (range_4h * 1.1 / 6)  # R1 level
    s1_4h = pivot_4h - (range_4h * 1.1 / 6)  # S1 level
    r1_4h_aligned = align_htf_to_ltf(prices, df_4h, r1_4h)
    s1_4h_aligned = align_htf_to_ltf(prices, df_4h, s1_4h)
    
    # Calculate 4h EMA34 for trend direction
    close_4h_series = pd.Series(df_4h['close'])
    ema_34_4h = close_4h_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d volume average (20-period)
    volume_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # 1d volume filter: current 1h volume > 1.5 x 20-period average 1d volume
    # Scale 1d average volume to 1h by dividing by 24 (approximate)
    volume_filter = volume > (vol_avg_1d_aligned / 24 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_exit = 0  # bars since last exit to prevent overtrading
    
    start_idx = max(34, 20)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        bars_since_exit += 1
        
        # Skip if any data is not ready
        if (np.isnan(r1_4h_aligned[i]) or np.isnan(s1_4h_aligned[i]) or 
            np.isnan(ema_34_4h_aligned[i]) or np.isnan(vol_avg_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            continue
        
        # Apply session filter: only trade during 08-20 UTC
        if not session_mask[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            continue
        
        if position == 0:
            # Minimum 24 bars between trades (1 day on 1h TF) to reduce frequency
            if bars_since_exit < 24:
                continue
                
            # Long: price breaks above R1 with 4h EMA34 uptrend and 1d volume spike
            if (close[i] > r1_4h_aligned[i] and close[i-1] <= r1_4h_aligned[i-1] and 
                close[i] > ema_34_4h_aligned[i] and volume_filter[i]):
                signals[i] = 0.20
                position = 1
                bars_since_exit = 0
            # Short: price breaks below S1 with 4h EMA34 downtrend and 1d volume spike
            elif (close[i] < s1_4h_aligned[i] and close[i-1] >= s1_4h_aligned[i-1] and 
                  close[i] < ema_34_4h_aligned[i] and volume_filter[i]):
                signals[i] = -0.20
                position = -1
                bars_since_exit = 0
        elif position != 0:
            # Exit: price returns to opposite EMA34 side (trend reversal)
            if position == 1 and close[i] < ema_34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            elif position == -1 and close[i] > ema_34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                # Hold position
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals