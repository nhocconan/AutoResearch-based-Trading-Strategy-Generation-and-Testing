#!/usr/bin/env python3
# [24982] 12h_1d_williams_alligator_v1
# Hypothesis: Williams Alligator on 1-day timeframe as trend filter, with price crossing Alligator's Jaw (13-period smoothed median) on 12-hour timeframe for entry. Uses volume confirmation (>1.5x average) to filter false breakouts. Works in both bull and bear markets by only taking trades in direction of Alligator alignment (all three lines ordered). Long when Jaw > Teeth > Lips and price crosses above Jaw; Short when Jaw < Teeth < Lips and price crosses below Jaw. Exit when price crosses back over Jaw or Alligator alignment breaks.
# Timeframe: 12h, HTF: 1d

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_williams_alligator_v1"
timeframe = "12h"
leverage = 1.0

def _sma(arr, window):
    """Simple moving average with NaN for insufficient data."""
    if window < 1:
        return arr.copy()
    res = np.full_like(arr, np.nan, dtype=float)
    for i in range(window - 1, len(arr)):
        res[i] = np.mean(arr[i - window + 1:i + 1])
    return res

def _smma(arr, window):
    """Smoothed moving average (SMMA) as used in Williams Alligator."""
    if window < 1 or len(arr) == 0:
        return np.full_like(arr, np.nan, dtype=float)
    res = np.full_like(arr, np.nan, dtype=float)
    # First value is simple average
    res[window - 1] = np.mean(arr[:window])
    # Subsequent values: SMMA = (PREV_SMMA * (N-1) + CURRENT_VALUE) / N
    for i in range(window, len(arr)):
        if not np.isnan(res[i - 1]):
            res[i] = (res[i - 1] * (window - 1) + arr[i]) / window
    return res

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-day data for Williams Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Williams Alligator components (13, 8, 5 periods with future shifts)
    # Alligator's Jaw: 13-period SMMA shifted by 8 bars
    # Alligator's Teeth: 8-period SMMA shifted by 5 bars
    # Alligator's Lips: 5-period SMMA shifted by 3 bars
    median_1d = (df_1d['high'].values + df_1d['low'].values) / 2.0
    
    jaw_raw = _smma(median_1d, 13)   # Jaw: 13-period SMMA
    teeth_raw = _smma(median_1d, 8)  # Teeth: 8-period SMMA
    lips_raw = _smma(median_1d, 5)   # Lips: 5-period SMMA
    
    # Apply the forward shifts (8, 5, 3 bars respectively)
    jaw = np.full_like(jaw_raw, np.nan)
    teeth = np.full_like(teeth_raw, np.nan)
    lips = np.full_like(lips_raw, np.nan)
    
    if len(jaw_raw) > 8:
        jaw[8:] = jaw_raw[:-8]
    if len(teeth_raw) > 5:
        teeth[5:] = teeth_raw[:-5]
    if len(lips_raw) > 3:
        lips[3:] = lips_raw[:-3]
    
    # Align Alligator components to 12-hour timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Calculate volume moving average (50-period) for confirmation
    vol_ma = _sma(volume, 50)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup period (need enough data for all indicators)
    start_idx = max(50, 13)  # Need at least 50 for volume MA, 13 for jaw
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price = close[i]
        
        # Check Alligator alignment: Jaw > Teeth > Lips for uptrend, Jaw < Teeth < Lips for downtrend
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        
        is_uptrend_aligned = jaw_val > teeth_val > lips_val
        is_downtrend_aligned = jaw_val < teeth_val < lips_val
        
        if position == 1:  # Long position
            # Exit conditions: price crosses below Jaw OR Alligator alignment breaks
            if price < jaw_val or not is_uptrend_aligned:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: price crosses above Jaw OR Alligator alignment breaks
            if price > jaw_val or not is_downtrend_aligned:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat - look for entry
            # Enter long: price crosses above Jaw with volume confirmation and uptrend alignment
            if price > jaw_val and vol_ratio > 1.5 and is_uptrend_aligned:
                position = 1
                signals[i] = 0.25
            # Enter short: price crosses below Jaw with volume confirmation and downtrend alignment
            elif price < jaw_val and vol_ratio > 1.5 and is_downtrend_aligned:
                position = -1
                signals[i] = -0.25
    
    return signals