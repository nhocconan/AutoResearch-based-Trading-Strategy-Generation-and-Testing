#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_VolumeSpike_Only
Hypothesis: Camarilla R3/S3 breakouts with volume spike (>2.0x 20-bar MA) and dynamic position sizing based on volume strength. Uses tighter breakout levels (R3/S3) for stronger momentum confirmation. Volume spike confirms institutional interest. Dynamic sizing (0.20-0.30) based on volume/z-score reduces whipsaws and overtrading. Target: 15-35 trades/year.
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
    
    # Previous 1d bar's OHLC for Camarilla levels (R3/S3 = stronger breakout levels)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_vals = df_1d['close'].values
    
    # Calculate Camarilla levels: R3, S3 (stronger breakout levels)
    rng = high_1d - low_1d
    camarilla_r3 = close_1d_vals + (rng * 1.1 / 4)   # R3 level
    camarilla_s3 = close_1d_vals - (rng * 1.1 / 4)   # S3 level
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: volume > 2.0x 20-period average (dynamic threshold)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # Volume z-score for dynamic position sizing (stronger volume = larger position)
    vol_std = pd.Series(volume).rolling(window=20, min_periods=20).std().values
    vol_zscore = np.where(vol_std > 0, (volume - vol_ma) / vol_std, 0)
    # Dynamic size: 0.20 base + 0.10 * min(zscore/2, 1.0) = range 0.20-0.30
    dynamic_size = 0.20 + 0.10 * np.minimum(np.maximum(vol_zscore / 2, 0), 1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of calculations (20 for vol)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(vol_zscore[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        camarilla_r3_val = camarilla_r3_aligned[i]
        camarilla_s3_val = camarilla_s3_aligned[i]
        vol_spike = volume_spike[i]
        size = dynamic_size[i]
        
        # Entry conditions: breakout of Camarilla R3/S3 with volume spike
        long_entry = (close_val > camarilla_r3_val) and vol_spike
        short_entry = (close_val < camarilla_s3_val) and vol_spike
        
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

name = "4h_Camarilla_R3_S3_Breakout_VolumeSpike_Only"
timeframe = "4h"
leverage = 1.0