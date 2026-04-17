#!/usr/bin/env python3
"""
Hypothesis: 6h strategy using 12h Supertrend (ATR=10, mult=3.0) for trend direction,
with 6h Donchian(20) breakout for entry timing and 6h volume > 1.5x 20-period MA for confirmation.
Long when: price breaks above Donchian upper band + volume spike + 12h Supertrend uptrend.
Short when: price breaks below Donchian lower band + volume spike + 12h Supertrend downtrend.
Exit when: price closes opposite the Donchian band (lower for long, upper for short) or Supertrend flips.
Fixed position size 0.25 to manage drawdown. Designed for 6h timeframe with strict entry
conditions to target 50-150 total trades over 4 years (12-37/year). Works in bull markets
(buying breakouts with uptrend) and bear markets (selling breakdowns with downtrend).
Uses proven edge: Supertrend for HTF trend filtering + Donchian breakouts + volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data for Donchian bands and ATR (primary timeframe)
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Calculate 6h ATR(10) for Supertrend
    tr1 = high_6h - low_6h
    tr2 = np.abs(high_6h - np.roll(close_6h, 1))
    tr3 = np.abs(low_6h - np.roll(close_6h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr_10 = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Calculate 6h Supertrend (ATR=10, mult=3.0)
    hl2 = (high_6h + low_6h) / 2.0
    upper_band = hl2 + 3.0 * atr_10
    lower_band = hl2 - 3.0 * atr_10
    
    supertrend = np.full_like(close_6h, np.nan, dtype=float)
    direction = np.full_like(close_6h, 1, dtype=int)  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(close_6h)):
        if np.isnan(atr_10[i]) or atr_10[i] == 0:
            supertrend[i] = supertrend[i-1] if not np.isnan(supertrend[i-1]) else hl2[i]
            direction[i] = direction[i-1]
            continue
            
        if close_6h[i-1] > upper_band[i-1]:
            direction[i] = 1
        elif close_6h[i-1] < lower_band[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
        
        if direction[i] == 1:
            lower_band[i] = max(lower_band[i], lower_band[i-1])
            supertrend[i] = lower_band[i]
        else:
            upper_band[i] = min(upper_band[i], upper_band[i-1])
            supertrend[i] = upper_band[i]
    
    # Get 12h data for Supertrend trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h ATR(10) for Supertrend
    tr1_12h = high_12h - low_12h
    tr2_12h = np.abs(high_12h - np.roll(close_12h, 1))
    tr3_12h = np.abs(low_12h - np.roll(close_12h, 1))
    tr_12h = np.maximum(tr1_12h, np.maximum(tr2_12h, tr3_12h))
    tr_12h[0] = tr1_12h[0]
    atr_10_12h = pd.Series(tr_12h).rolling(window=10, min_periods=10).mean().values
    
    # Calculate 12h Supertrend (ATR=10, mult=3.0)
    hl2_12h = (high_12h + low_12h) / 2.0
    upper_band_12h = hl2_12h + 3.0 * atr_10_12h
    lower_band_12h = hl2_12h - 3.0 * atr_10_12h
    
    supertrend_12h = np.full_like(close_12h, np.nan, dtype=float)
    direction_12h = np.full_like(close_12h, 1, dtype=int)
    
    for i in range(1, len(close_12h)):
        if np.isnan(atr_10_12h[i]) or atr_10_12h[i] == 0:
            supertrend_12h[i] = supertrend_12h[i-1] if not np.isnan(supertrend_12h[i-1]) else hl2_12h[i]
            direction_12h[i] = direction_12h[i-1]
            continue
            
        if close_12h[i-1] > upper_band_12h[i-1]:
            direction_12h[i] = 1
        elif close_12h[i-1] < lower_band_12h[i-1]:
            direction_12h[i] = -1
        else:
            direction_12h[i] = direction_12h[i-1]
        
        if direction_12h[i] == 1:
            lower_band_12h[i] = max(lower_band_12h[i], lower_band_12h[i-1])
            supertrend_12h[i] = lower_band_12h[i]
        else:
            upper_band_12h[i] = min(upper_band_12h[i], upper_band_12h[i-1])
            supertrend_12h[i] = upper_band_12h[i]
    
    # Calculate 6h Donchian bands (20-period)
    highest_high_20 = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # Volume average (20-period) on 6h for confirmation
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align all HTF indicators to primary timeframe (6h)
    supertrend_12h_aligned = align_htf_to_ltf(prices, df_12h, supertrend_12h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or 
            np.isnan(supertrend_12h_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        donchian_upper = highest_high_20[i]
        donchian_lower = lowest_low_20[i]
        supertrend_dir = supertrend_12h_aligned[i]
        vol_ma = volume_ma_20[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Look for breakouts with volume confirmation and 12h Supertrend filter
            # Long: price breaks above Donchian upper + volume spike + 12h Supertrend uptrend
            if price > donchian_upper and vol > 1.5 * vol_ma and supertrend_dir > 0:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower + volume spike + 12h Supertrend downtrend
            elif price < donchian_lower and vol > 1.5 * vol_ma and supertrend_dir < 0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit on close below Donchian lower or 12h Supertrend flips to downtrend
            if price < donchian_lower or supertrend_dir < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit on close above Donchian upper or 12h Supertrend flips to uptrend
            if price > donchian_upper or supertrend_dir > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Supertrend12h_Donchian20_VolumeSpike"
timeframe = "6h"
leverage = 1.0