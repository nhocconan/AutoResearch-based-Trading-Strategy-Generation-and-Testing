#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 breakout with 1w EMA34 trend filter and volume confirmation.
- Camarilla levels: R3 = close + 1.1*(high-low)/4, S3 = close - 1.1*(high-low)/4
- Long: price breaks above R3 + price > 1w EMA34 (uptrend) + volume > 2.0x 24-period avg
- Short: price breaks below S3 + price < 1w EMA34 (downtrend) + volume > 2.0x 24-period avg
- Exit: price crosses 1w EMA34 (trend-based exit)
- Uses weekly EMA34 for stronger trend filter (less whipsaw in bear markets)
- Volume confirmation ensures breakout validity
- Target: 75-200 total trades over 4 years (19-50/year) on 4h timeframe
- Discrete position sizing: ±0.25 to minimize fee churn
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
    
    # Volume confirmation: > 2.0x 24-period average (strict spike filter)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Load 1w data ONCE before loop for EMA34 trend filter and Camarilla calculation
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate 1w EMA34 for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Camarilla levels from previous 1w bar (R3, S3)
    # R3 = close + 1.1*(high-low)/4, S3 = close - 1.1*(high-low)/4
    camarilla_r3 = close_1w + 1.1 * (high_1w - low_1w) / 4
    camarilla_s3 = close_1w - 1.1 * (high_1w - low_1w) / 4
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(24, 34)  # Need 24 for volume MA, 34 for 1w EMA34
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(ema_34_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike confirmation (> 2.0x average)
        volume_spike = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above R3 + price > 1w EMA34 (uptrend) + volume spike
            if volume_spike and close[i] > camarilla_r3_aligned[i] and close[i] > ema_34_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 + price < 1w EMA34 (downtrend) + volume spike
            elif volume_spike and close[i] < camarilla_s3_aligned[i] and close[i] < ema_34_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below 1w EMA34 (trend-based exit)
            if close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above 1w EMA34 (trend-based exit)
            if close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1wEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0