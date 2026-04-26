#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_v1
Hypothesis: On 12h timeframe, Camarilla R3/S3 breakouts with 1d EMA34 trend filter and volume spike produce high-probability trades in both bull and bear markets. The 1d EMA34 establishes the primary trend, while Camarilla R3/S3 levels provide robust breakout entries less prone to false signals than R1/S1. Volume confirmation ensures breakout validity. Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for HTF trend filter (EMA34) and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d Camarilla levels (R3, S3) - more robust than R1/S1
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_shifted = np.concatenate([[np.nan], close_1d[:-1]])  # previous 1d close
    
    # Camarilla calculation uses previous 1d's OHLC
    camarilla_range = high_1d - low_1d
    r3 = close_1d_shifted + 1.1 * camarilla_range / 4  # R3 level
    s3 = close_1d_shifted - 1.1 * camarilla_range / 4  # S3 level
    
    # Align Camarilla levels to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # 12h volume confirmation: volume > 2.0x 20-period average (stricter for fewer trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for EMA, 20 for volume MA)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # 1d trend filter (EMA34)
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Volume confirmation (stricter threshold)
        volume_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        # Camarilla breakout conditions (R3/S3)
        breakout_r3 = close[i] > r3_aligned[i]
        breakout_s3 = close[i] < s3_aligned[i]
        
        # Long logic: breakout above R3 in uptrend with volume spike
        if uptrend and volume_spike and breakout_r3:
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        # Short logic: breakout below S3 in downtrend with volume spike
        elif downtrend and volume_spike and breakout_s3:
            if position != -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        # Exit conditions: loss of trend (we let winners run until trend changes)
        elif position == 1 and not uptrend:
            signals[i] = 0.0
            position = 0
        elif position == -1 and not downtrend:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0