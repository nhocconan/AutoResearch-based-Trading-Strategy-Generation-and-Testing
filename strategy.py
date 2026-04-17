#!/usr/bin/env python3
"""
12h_Williams_Alligator_Range_Bound_v1
Williams Alligator with range-bound filter for 12h timeframe.
Uses Alligator (SMAs with offset) to detect trending vs ranging markets.
In ranging markets (JAW > TEETH > LIPS or LIPS > TEETH > JAW), fades extremes at
Williams %R overbought/oversold levels. In trending markets, follows Alligator
direction. Volume confirmation filters false signals.
Designed to work in both bull and bear markets by adapting to market regime.
Target: 50-150 total trades over 4 years (12-37/year).
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
    
    # === 1d Williams Alligator (SMAs with offset) ===
    df_1d = get_htf_data(prices, '1d')
    median_price_1d = (df_1d['high'].values + df_1d['low'].values) / 2
    
    # Jaw: 13-period SMA, shifted 8 bars
    jaw = np.full_like(median_price_1d, np.nan)
    for i in range(len(median_price_1d)):
        if i >= 13:
            jaw[i] = np.mean(median_price_1d[i-12:i+1])
    jaw = np.roll(jaw, 8)  # shift 8 bars forward
    
    # Teeth: 8-period SMA, shifted 5 bars
    teeth = np.full_like(median_price_1d, np.nan)
    for i in range(len(median_price_1d)):
        if i >= 8:
            teeth[i] = np.mean(median_price_1d[i-7:i+1])
    teeth = np.roll(teeth, 5)  # shift 5 bars forward
    
    # Lips: 5-period SMA, shifted 3 bars
    lips = np.full_like(median_price_1d, np.nan)
    for i in range(len(median_price_1d)):
        if i >= 5:
            lips[i] = np.mean(median_price_1d[i-4:i+1])
    lips = np.roll(lips, 3)  # shift 3 bars forward
    
    # === 12h Williams %R (14-period) ===
    highest_high = np.full_like(high, np.nan)
    lowest_low = np.full_like(low, np.nan)
    for i in range(len(high)):
        if i >= 14:
            highest_high[i] = np.max(high[i-13:i+1])
            lowest_low[i] = np.min(low[i-13:i+1])
        else:
            highest_high[i] = np.max(high[:i+1]) if i > 0 else high[i]
            lowest_low[i] = np.min(low[:i+1]) if i > 0 else low[i]
    
    williams_r = np.where(
        (highest_high - lowest_low) != 0,
        -100 * ((highest_high - close) / (highest_high - lowest_low)),
        -50
    )
    
    # === 12h Volume confirmation (20-period average) ===
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(len(volume)):
        if i >= 20:
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
        elif i > 0:
            vol_ma_20[i] = np.mean(volume[max(0, i-9):i+1])
        else:
            vol_ma_20[i] = volume[0]
    
    vol_confirm = volume > vol_ma_20 * 1.5  # volume spike: 1.5x average
    
    # === Align 1d indicators to 12h timeframe ===
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_confirm[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Determine market regime using Alligator alignment
        # Trending up: Lips > Teeth > Jaw
        # Trending down: Jaw > Teeth > Lips
        # Ranging: otherwise (JAW > TEETH > LIPS or LIPS > TEETH > JAW)
        lips_val = lips_aligned[i]
        teeth_val = teeth_aligned[i]
        jaw_val = jaw_aligned[i]
        
        is_trending_up = (lips_val > teeth_val) and (teeth_val > jaw_val)
        is_trending_down = (jaw_val > teeth_val) and (teeth_val > lips_val)
        is_ranging = not (is_trending_up or is_trending_down)
        
        # Entry logic: only enter when flat
        if position == 0:
            if is_ranging:
                # In ranging market: fade extremes
                if williams_r[i] < -80 and vol_confirm[i]:  # oversold
                    signals[i] = 0.25
                    position = 1
                    continue
                elif williams_r[i] > -20 and vol_confirm[i]:  # overbought
                    signals[i] = -0.25
                    position = -1
                    continue
            else:
                # In trending market: follow direction
                if is_trending_up and vol_confirm[i]:
                    signals[i] = 0.25
                    position = 1
                    continue
                elif is_trending_down and vol_confirm[i]:
                    signals[i] = -0.25
                    position = -1
                    continue
        
        # Exit logic
        elif position == 1:
            # Exit long: Williams %R overbought OR Alligator signals trend change
            if williams_r[i] > -20 or not is_trending_up:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R oversold OR Alligator signals trend change
            if williams_r[i] < -80 or not is_trending_down:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Williams_Alligator_Range_Bound_v1"
timeframe = "12h"
leverage = 1.0