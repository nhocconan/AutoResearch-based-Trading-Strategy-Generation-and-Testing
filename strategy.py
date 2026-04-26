#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_v1
Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation.
In bull markets: price breaks above R3 + above 1d EMA34 + volume spike = long.
In bear markets: price breaks below S3 + below 1d EMA34 + volume spike = short.
The 1d EMA34 acts as a higher timeframe trend filter to avoid counter-trend whipsaws.
Volume confirmation ensures breakout conviction. Discrete sizing (0.30) limits fee drag.
Target: 100-180 total trades over 4 years (25-45/year) by requiring R3/S3 breakout, trend alignment, and volume spike.
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
    
    # Load 1d data ONCE before loop for HTF EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous 1d bar
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), etc.
    # We need previous day's high, low, close
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar (using that day's HLC)
    camarilla_R3 = close_1d + 1.1 * (high_1d - low_1d)
    camarilla_S3 = close_1d - 1.1 * (high_1d - low_1d)
    
    # Align Camarilla levels to 4h timeframe (completed 1d bars only)
    R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3)
    
    # 4h volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for EMA, 20 for volume MA)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(R3_aligned[i]) or
            np.isnan(S3_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.30
            else:
                signals[i] = -0.30
            continue
        
        # Volume spike condition
        volume_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        # Camarilla breakout conditions
        breakout_up = close[i] > R3_aligned[i]   # Price breaks above R3
        breakout_down = close[i] < S3_aligned[i]  # Price breaks below S3
        
        # 1d EMA34 trend filter
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        if breakout_up and uptrend and volume_spike:
            # Long signal: break above R3 + uptrend + volume spike
            if position != 1:
                signals[i] = 0.30
                position = 1
            else:
                signals[i] = 0.30
        elif breakout_down and downtrend and volume_spike:
            # Short signal: break below S3 + downtrend + volume spike
            if position != -1:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = -0.30
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.30
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0