#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike
Hypothesis: Use 1d EMA34 for trend direction, 1d volume spike (>2x avg) for confirmation,
and 4h timeframe for precise entry at Camarilla R3/S3 breakouts.
Targets 20-40 trades/year by requiring 1d trend alignment and volume confirmation.
Position size 0.25 manages drawdown. Works in bull/bear via trend filter + volume exhaustion logic.
"""

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
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
    
    # Get 1d data for trend, volume, and Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d average volume for volume filter (20-period average)
    vol_avg_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # Calculate Camarilla levels from previous 1d bar (R3, S3)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla R3 = C + (H-L) * 1.1/6
    # Camarilla S3 = C - (H-L) * 1.1/6
    rng = prev_high - prev_low
    r3 = prev_close + (rng * 1.1 / 6)
    s3 = prev_close - (rng * 1.1 / 6)
    
    # Align 1d levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 1d EMA34 (34) and 1d vol avg (20)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_avg_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Higher timeframe trend filter (1d)
        uptrend_1d = close[i] > ema_34_1d_aligned[i]
        downtrend_1d = close[i] < ema_34_1d_aligned[i]
        
        # Volume filter: current 4h volume > 2x average 1d volume
        vol_4h = volume[i]
        vol_avg = vol_avg_1d_aligned[i]
        volume_filter = vol_4h > vol_avg * 2.0
        
        if position == 0:
            # Long entry: uptrend + price breaks above R3 + volume filter
            if uptrend_1d and close[i] > r3_aligned[i] and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short entry: downtrend + price breaks below S3 + volume filter
            elif downtrend_1d and close[i] < s3_aligned[i] and volume_filter:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend breaks or price re-enters below R3
            if not uptrend_1d or close[i] < r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend breaks or price re-enters above S3
            if not downtrend_1d or close[i] > s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals