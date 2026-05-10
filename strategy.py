#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolFilter
Hypothesis: Use 4h trend and 1d volume filter for direction, 1h for precise entry at Camarilla R1/S1 breakouts.
Targets 20-40 trades/year by requiring 4h EMA50 trend alignment and 1d volume spike (>2x avg). 
Position size 0.20 manages drawdown. Works in bull/bear via trend filter + volume exhaustion logic.
"""

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolFilter"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Get 1d data for Camarilla pivots and volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1d average volume for volume filter
    vol_avg_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # Calculate Camarilla levels from previous 1d bar (R1, S1)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla R1 = C + (H-L) * 1.1/12
    # Camarilla S1 = C - (H-L) * 1.1/12
    rng = prev_high - prev_low
    r1 = prev_close + (rng * 1.1 / 12)
    s1 = prev_close - (rng * 1.1 / 12)
    
    # Align 1d levels to 1h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 4h EMA50 (50) and 1d vol avg (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(vol_avg_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Higher timeframe trend filter (4h)
        uptrend_4h = close[i] > ema_50_4h_aligned[i]
        downtrend_4h = close[i] < ema_50_4h_aligned[i]
        
        # Volume filter: current 1h volume > 2x average 1d volume
        # This captures institutional interest during breakouts
        vol_1h = volume[i]
        vol_avg = vol_avg_1d_aligned[i]
        volume_filter = vol_1h > vol_avg * 2.0
        
        if position == 0:
            # Long entry: uptrend + price breaks above R1 + volume filter
            if uptrend_4h and close[i] > r1_aligned[i] and volume_filter:
                signals[i] = 0.20
                position = 1
            # Short entry: downtrend + price breaks below S1 + volume filter
            elif downtrend_4h and close[i] < s1_aligned[i] and volume_filter:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: trend breaks or price re-enters below R1
            if not uptrend_4h or close[i] < r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: trend breaks or price re-enters above S1
            if not downtrend_4h or close[i] > s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals