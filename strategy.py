# %pip install git+https://github.com/beanlouie/ta-lib.git

#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_1wVolFilter
Hypothesis: Use 1d trend and 1w volume filter for direction, 12h for precise entry at Camarilla R1/S1 breakouts.
Targets 15-30 trades/year by requiring 1d EMA50 trend alignment and 1w volume spike (>1.5x avg). 
Position size 0.25 manages drawdown. Works in bull/bear via trend filter + volume exhaustion logic.
"""

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_1wVolFilter"
timeframe = "12h"
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
    
    # Get 1d data for trend filter and Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 1w data for volume filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1w average volume for volume filter
    vol_avg_1w = pd.Series(df_1w['volume']).rolling(window=10, min_periods=10).mean().values
    vol_avg_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_avg_1w)
    
    # Calculate Camarilla levels from previous 1d bar (R1, S1)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla R1 = C + (H-L) * 1.1/12
    # Camarilla S1 = C - (H-L) * 1.1/12
    rng = prev_high - prev_low
    r1 = prev_close + (rng * 1.1 / 12)
    s1 = prev_close - (rng * 1.1 / 12)
    
    # Align 1d levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 1d EMA50 (50) and 1w vol avg (10)
    start_idx = max(50, 10)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_avg_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Higher timeframe trend filter (1d)
        uptrend_1d = close[i] > ema_50_1d_aligned[i]
        downtrend_1d = close[i] < ema_50_1d_aligned[i]
        
        # Volume filter: current 12h volume > 1.5x average 1w volume
        # This captures institutional interest during breakouts
        vol_12h = volume[i]
        vol_avg = vol_avg_1w_aligned[i]
        volume_filter = vol_12h > vol_avg * 1.5
        
        if position == 0:
            # Long entry: uptrend + price breaks above R1 + volume filter
            if uptrend_1d and close[i] > r1_aligned[i] and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short entry: downtrend + price breaks below S1 + volume filter
            elif downtrend_1d and close[i] < s1_aligned[i] and volume_filter:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend breaks or price re-enters below R1
            if not uptrend_1d or close[i] < r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend breaks or price re-enters above S1
            if not downtrend_1d or close[i] > s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals