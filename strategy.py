#!/usr/bin/env python3
"""
6h_Donchian20_WeeklyPivot_Direction_VolumeFilter_V1
Hypothesis: 6h Donchian(20) breakout in direction of weekly Camarilla pivot (R4/S4 from 1w HTF) with volume confirmation (>1.3x 20-period volume MA). 
Weekly pivot provides institutional bias (R4/S4 as strong break/flip levels). Volume spike confirms participation. 
In bear markets, weekly S4 acts as strong resistance for shorts; in bull markets, R4 as support for longs. 
Target 12-37 trades/year (50-150 total over 4 years) via tight entry conditions.
Uses 6h primary timeframe with 1w HTF for weekly pivot bias.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for weekly Camarilla pivot)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # === 1w Camarilla Pivot Levels (R4, S4) ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Typical price for pivot calculation
    typical_price = (high_1w + low_1w + close_1w) / 3.0
    range_1w = high_1w - low_1w
    
    # Weekly Camarilla levels (R4 and S4 are strongest)
    camarilla_r4_1w = close_1w + (range_1w * 1.1 / 2.0)  # R4 = close + range*1.1/2
    camarilla_s4_1w = close_1w - (range_1w * 1.1 / 2.0)  # S4 = close - range*1.1/2
    
    # Align weekly Camarilla levels to 6h timeframe
    camarilla_r4_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r4_1w)
    camarilla_s4_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s4_1w)
    
    # === 6h Indicators (primary timeframe) ===
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # Volume MA (20-period) for spike detection
    vol_ma = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if indicators not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) 
            or np.isnan(camarilla_r4_1w_aligned[i]) or np.isnan(camarilla_s4_1w_aligned[i])
            or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_6h[i]
        vol = volume_6h[i]
        vol_ok = vol > 1.3 * vol_ma[i]  # volume spike confirmation
        
        if position == 0:
            # Long: price breaks above Donchian high + volume spike + weekly R4 support (price > R4)
            if price > donchian_high[i] and vol_ok and price > camarilla_r4_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low + volume spike + weekly S4 resistance (price < S4)
            elif price < donchian_low[i] and vol_ok and price < camarilla_s4_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below Donchian low (reversal) or volume fails
            if price < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above Donchian high (reversal) or volume fails
            if price > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_WeeklyPivot_Direction_VolumeFilter_V1"
timeframe = "6h"
leverage = 1.0