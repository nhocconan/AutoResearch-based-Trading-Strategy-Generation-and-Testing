#!/usr/bin/env python3
"""
1d_Camarilla_R3S3_Breakout_1wTrend_VolumeSpike
Hypothesis: Use weekly trend filter (EMA34) with daily Camarilla R3/S3 breakouts and volume confirmation.
Long when price breaks above R3 in weekly uptrend with volume spike; short when breaks below S3 in weekly downtrend.
Target: 15-25 trades/year (~60-100 total over 4 years) to minimize fee drag and avoid overtrading.
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
    
    # Get daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate daily Camarilla pivot levels (R3, S3)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point
    pivot = (high_1d + low_1d + close_1d) / 3.0
    # Daily range
    range_1d = high_1d - low_1d
    # Camarilla levels
    R3 = pivot + range_1d * 1.1 / 4.0
    S3 = pivot - range_1d * 1.1 / 4.0
    
    # Align daily Camarilla levels to 1d timeframe (they're already aligned)
    R3_aligned = R3
    S3_aligned = S3
    
    # Calculate volume ratio (current / 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need weekly EMA34 (34) and volume MA (20)
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or
            np.isnan(vol_ratio[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Trend filter: weekly EMA34 slope
        weekly_uptrend = ema_34_1w_aligned[i] > ema_34_1w_aligned[i-1]
        weekly_downtrend = ema_34_1w_aligned[i] < ema_34_1w_aligned[i-1]
        
        # Volume confirmation: significant spike
        volume_spike = vol_ratio[i] > 2.0
        
        if position == 0:
            # Long: price breaks above R3 in weekly uptrend with volume spike
            if close[i] > R3_aligned[i] and weekly_uptrend and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 in weekly downtrend with volume spike
            elif close[i] < S3_aligned[i] and weekly_downtrend and volume_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price crosses below pivot OR weekly trend turns down
            if close[i] < pivot[i] or not weekly_uptrend:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price crosses above pivot OR weekly trend turns up
            if close[i] > pivot[i] or not weekly_downtrend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_R3S3_Breakout_1wTrend_VolumeSpike"
timeframe = "1d"
leverage = 1.0