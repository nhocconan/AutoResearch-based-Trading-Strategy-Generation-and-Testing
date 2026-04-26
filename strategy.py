#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dEMA34_VolumeSpike_v2
Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume spike confirmation. 
Long: price breaks above R1 + above 1d EMA34 + volume spike.
Short: price breaks below S1 + below 1d EMA34 + volume spike.
Uses tighter R1/S1 levels for more frequent but still filtered signals. Target: 100-180 total trades over 4 years (25-45/year).
Uses discrete position sizing (0.30) to limit fee drag. Works in bull/bear via trend filter.
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
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla: R1 = close + 1.09*(high-low), S1 = close - 1.09*(high-low)
    camarilla_R1 = close_1d + 1.09 * (high_1d - low_1d)
    camarilla_S1 = close_1d - 1.09 * (high_1d - low_1d)
    
    # Align Camarilla levels to 4h timeframe (completed 1d bars only)
    R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1)
    
    # 4h volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for EMA, 20 for volume MA)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(R1_aligned[i]) or
            np.isnan(S1_aligned[i]) or
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
        breakout_up = close[i] > R1_aligned[i]   # Price breaks above R1
        breakout_down = close[i] < S1_aligned[i]  # Price breaks below S1
        
        # 1d EMA34 trend filter
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        if breakout_up and uptrend and volume_spike:
            # Long signal: break above R1 + uptrend + volume spike
            if position != 1:
                signals[i] = 0.30
                position = 1
            else:
                signals[i] = 0.30
        elif breakout_down and downtrend and volume_spike:
            # Short signal: break below S1 + downtrend + volume spike
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

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_VolumeSpike_v2"
timeframe = "4h"
leverage = 1.0