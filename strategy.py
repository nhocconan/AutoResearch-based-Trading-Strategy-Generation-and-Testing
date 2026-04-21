#!/usr/bin/env python3
"""
6h_Camarilla_R1S1_Breakout_VolumeSpike_ATRFilter_V1
Hypothesis: 6h Camarilla pivot breakouts at R1/S1 with volume spike confirmation (>1.5x 20-period volume MA) and ATR-based trend filter (price > EMA34 for longs, < for shorts). 
Camarilla levels from 1d HTF provide institutional support/resistance. Volume spikes confirm institutional participation. 
EMA34 trend filter ensures alignment with medium-term trend. Target 12-37 trades/year (50-150 total over 4 years).
Uses 6h primary timeframe with 1d HTF for Camarilla calculation and EMA trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for Camarilla pivots and EMA trend)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # === 1d Camarilla Pivot Levels (R1, S1) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Typical price for pivot calculation
    typical_price = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    camarilla_r1 = close_1d + (range_1d * 1.1 / 12.0)
    camarilla_s1 = close_1d - (range_1d * 1.1 / 12.0)
    
    # Align Camarilla levels to 6h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # === 1d EMA34 for trend filter ===
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === 6h Indicators (primary timeframe) ===
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # Volume MA (20-period) for spike detection
    vol_ma = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):
        # Skip if indicators not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) 
            or np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_6h[i]
        vol = volume_6h[i]
        vol_ok = vol > 1.5 * vol_ma[i]  # volume spike confirmation
        
        if position == 0:
            # Long: price breaks above R1 + volume spike + uptrend (price > EMA34)
            if price > camarilla_r1_aligned[i] and vol_ok and price > ema_34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + volume spike + downtrend (price < EMA34)
            elif price < camarilla_s1_aligned[i] and vol_ok and price < ema_34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below EMA34 or volume spike fails
            if price < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above EMA34 or volume spike fails
            if price > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R1S1_Breakout_VolumeSpike_ATRFilter_V1"
timeframe = "6h"
leverage = 1.0