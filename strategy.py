#!/usr/bin/env python3
"""
12h_1D_Camarilla_R3_S3_Breakout_VolumeSpike
Hypothesis: Price breakout from daily Camarilla R3/S3 levels with volume confirmation and 12h momentum filter.
Goes long when price breaks above daily R3 with volume > 2x average and 12h momentum positive.
Goes short when price breaks below daily S3 with volume > 2x average and 12h momentum negative.
Exits when price returns to daily central pivot (PP).
Designed to capture strong intraday moves with confirmation, avoiding false breakouts.
Target: 15-25 trades/year (60-100 total over 4 years) to minimize fee drag while capturing major moves.
Works in both bull and bear markets due to directional flexibility and volume confirmation.
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
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels: R3, S3, PP
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla formulas
    R3 = high_1d + (high_1d - low_1d) * 1.1 / 2
    S3 = low_1d - (high_1d - low_1d) * 1.1 / 2
    PP = (high_1d + low_1d + close_1d) / 3
    
    # Align daily levels to 12h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    PP_aligned = align_htf_to_ltf(prices, df_1d, PP)
    
    # Calculate 12h momentum (ROC 2-period)
    roc_period = 2
    roc = np.full(n, np.nan)
    if n > roc_period:
        roc[roc_period:] = (close[roc_period:] - close[:-roc_period]) / close[:-roc_period] * 100
    
    # Calculate volume average (20-period)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup: need all indicators
    start_idx = max(20, roc_period)
    
    for i in range(start_idx, n):
        if (np.isnan(R3_aligned[i]) or
            np.isnan(S3_aligned[i]) or
            np.isnan(PP_aligned[i]) or
            np.isnan(vol_ma_20[i]) or
            np.isnan(roc[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma_20[i] if vol_ma_20[i] > 0 else 0
        
        # Momentum filter: positive for long, negative for short
        mom_positive = roc[i] > 0
        mom_negative = roc[i] < 0
        
        # Volume confirmation: > 2x average volume
        volume_confirmation = vol_ratio > 2.0
        
        if position == 0:
            # Long: break above daily R3 with volume and positive momentum
            if mom_positive and volume_confirmation and price > R3_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below daily S3 with volume and negative momentum
            elif mom_negative and volume_confirmation and price < S3_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: return to daily central pivot (PP)
            if price < PP_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # Maintain position
        elif position == -1:
            # Short exit: return to daily central pivot (PP)
            if price > PP_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # Maintain position
    
    return signals

name = "12h_1D_Camarilla_R3_S3_Breakout_VolumeSpike"
timeframe = "12h"
leverage = 1.0