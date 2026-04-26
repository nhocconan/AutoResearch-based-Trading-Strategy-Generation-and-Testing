#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike
Hypothesis: On 12h timeframe, Camarilla R3/S3 breakouts with 1d EMA34 trend filter and volume spike confirmation. 
Uses tighter R3/S3 levels for stronger breakouts. Volume spike ensures participation, 1d EMA34 filter ensures 
trading with higher timeframe trend. Fixed size 0.25 to limit trades. Target: 12-37 trades/year.
Works in bull/bear: In bull markets, breakouts with trend continuation work; in bear markets, 
short breakouts against the trend (but aligned with 1d EMA34) capture mean reversion moves.
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
    
    # Load 1d data ONCE before loop for HTF filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Previous 12h bar's OHLC for Camarilla levels (R3/S3 = stronger breakout levels)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h_vals = df_12h['close'].values
    
    # Calculate Camarilla levels: R3, S3 (stronger breakout levels)
    rng = high_12h - low_12h
    camarilla_r3 = close_12h_vals + (rng * 1.1 / 4)   # R3 level
    camarilla_s3 = close_12h_vals - (rng * 1.1 / 4)   # S3 level
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    
    # 1d EMA34 trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike: volume > 2.0 * 20-period median volume (avoid percentile look-ahead)
    vol_median_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    volume_spike = volume > (2.0 * vol_median_20)
    
    # Fixed position size to control trade frequency
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of calculations (50 for 12h data, 34 for EMA, 20 for volume median)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_median_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        camarilla_r3_val = camarilla_r3_aligned[i]
        camarilla_s3_val = camarilla_s3_aligned[i]
        ema_34_val = ema_34_1d_aligned[i]
        vol_spike = volume_spike[i]
        size = fixed_size
        
        # Entry conditions: breakout of Camarilla R3/S3 with volume spike AND 1d EMA34 trend alignment
        long_entry = (close_val > camarilla_r3_val) and vol_spike and (close_val > ema_34_val)
        short_entry = (close_val < camarilla_s3_val) and vol_spike and (close_val < ema_34_val)
        
        if position == 0:
            # Flat - look for entry
            if long_entry:
                signals[i] = size
                position = 1
            elif short_entry:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on mean reversion to midpoint (Camarilla center)
            mid_point = (camarilla_r3_val + camarilla_s3_val) / 2
            if close_val < mid_point:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on mean reversion to midpoint (Camarilla center)
            mid_point = (camarilla_r3_val + camarilla_s3_val) / 2
            if close_val > mid_point:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0