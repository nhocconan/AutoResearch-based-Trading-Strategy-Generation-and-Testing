#!/usr/bin/env python3
# 6h_WilliamsAlligator_Trend_Filter_12hVolume
# Hypothesis: Williams Alligator (SMAs with offset) on 6h defines trend, combined with 12h volume confirmation to avoid whipsaws. Works in bull/bear by filtering trend direction. Target: 20-40 trades/year.

timeframe = "6h"
name = "6h_WilliamsAlligator_Trend_Filter_12hVolume"
leverage = 1.0

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
    
    # Get 6h data for Williams Alligator
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) == 0:
        return np.zeros(n)
    
    # Williams Alligator: Jaw (13-period SMA, 8-bar offset), Teeth (8-period SMA, 5-bar offset), Lips (5-period SMA, 3-bar offset)
    close_6h = df_6h['close'].values
    jaw_raw = pd.Series(close_6h).rolling(window=13, min_periods=13).mean().values
    teeth_raw = pd.Series(close_6h).rolling(window=8, min_periods=8).mean().values
    lips_raw = pd.Series(close_6h).rolling(window=5, min_periods=5).mean().values
    
    # Apply offsets: Jaw shifted by 8, Teeth by 5, Lips by 3
    jaw = np.roll(jaw_raw, 8)
    teeth = np.roll(teeth_raw, 5)
    lips = np.roll(lips_raw, 3)
    # Invalidate the first N values after roll
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    jaw_aligned = align_htf_to_ltf(prices, df_6h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_6h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_6h, lips)
    
    # Get 12h data for volume confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) == 0:
        return np.zeros(n)
    
    vol_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(vol_12h).rolling(window=24, min_periods=24).mean().values  # 24 * 12h = 12 days
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(13, 24)  # Ensure we have Alligator and volume MA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(vol_ma_12h_aligned[i]) or vol_ma_12h_aligned[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (bullish alignment) and 12h volume above average
            if lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i] and volume[i] > vol_ma_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Lips < Teeth < Jaw (bearish alignment) and 12h volume above average
            elif lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i] and volume[i] > vol_ma_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Alligator alignment breaks (Lips < Teeth or Teeth < Jaw)
            if lips_aligned[i] < teeth_aligned[i] or teeth_aligned[i] < jaw_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Alligator alignment breaks (Lips > Teeth or Teeth > Jaw)
            if lips_aligned[i] > teeth_aligned[i] or teeth_aligned[i] > jaw_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals