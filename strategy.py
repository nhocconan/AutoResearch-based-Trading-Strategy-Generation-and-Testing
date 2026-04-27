#!/usr/bin/env python3
"""
4h Williams Alligator + Volume Spike + Trend Filter
Long when: 1) Price above Alligator teeth (green line), 2) Lips above teeth (bullish alignment), 3) Volume > 1.5x 20-period average.
Short when: 1) Price below Alligator teeth, 2) Lips below teeth (bearish alignment), 3) Volume > 1.5x 20-period average.
Exit when price crosses Alligator jaws (red line) or alignment breaks.
Williams Alligator identifies trends via smoothed medians; effective in both bull and bear markets when combined with volume confirmation.
Designed for 4h timeframe: targets 75-200 total trades over 4 years (19-50/year).
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
    
    # Get 4h data for Williams Alligator (13,8,5 SMAs of median price)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 13:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    median_4h = (high_4h + low_4h) / 2.0
    
    # Williams Alligator lines: Jaw (13), Teeth (8), Lips (5) - all SMAs of median
    jaw = pd.Series(median_4h).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(median_4h).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(median_4h).rolling(window=5, min_periods=5).mean().values
    
    # Align to 4h timeframe (no additional delay needed for SMAs)
    jaw_aligned = align_htf_to_ltf(prices, df_4h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_4h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_4h, lips)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma_20 = np.full(n, np.nan, dtype=np.float64)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Alligator (13 periods), volume MA (20 periods)
    start_idx = max(13, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current values
        price = close[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: volume > 1.5x average
        vol_filter = vol_now > 1.5 * vol_avg
        
        # Alligator alignment: Lips > Teeth > Jaw = bullish, Lips < Teeth < Jaw = bearish
        bullish_alignment = lips_val > teeth_val and teeth_val > jaw_val
        bearish_alignment = lips_val < teeth_val and teeth_val < jaw_val
        
        if position == 0:
            # Long: price above teeth + bullish alignment + volume spike
            if price > teeth_val and bullish_alignment and vol_filter:
                signals[i] = size
                position = 1
            # Short: price below teeth + bearish alignment + volume spike
            elif price < teeth_val and bearish_alignment and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses jaws (red line) or alignment breaks
            if price < jaw_val or not bullish_alignment:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses jaws (red line) or alignment breaks
            if price > jaw_val or not bearish_alignment:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Williams_Alligator_Volume_Spike_Trend"
timeframe = "4h"
leverage = 1.0