#!/usr/bin/env python3
"""
12h Williams Alligator + Volume Spike + 1d Chop Regime Filter
Hypothesis: Williams Alligator identifies trending vs ranging markets. 
- Jaw (13-period SMMA shifted 8), Teeth (8-period SMMA shifted 5), Lips (5-period SMMA shifted 3)
- Trending: Lips > Teeth > Jaw (bull) or Lips < Teeth < Jaw (bear)
- Ranging: lines intertwined
Only trade in trending regime with volume confirmation. 1d chop filter avoids whipsaws in sideways markets.
Works in bull via trend-following breaks and in bear via short signals aligned with daily trend filter.
Target: 12-30 trades/year on 12h (50-120 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(source, period):
    """Smoothed Moving Average (SMMA) aka Wilder's smoothing"""
    if len(source) < period:
        return np.full_like(source, np.nan, dtype=float)
    result = np.empty_like(source)
    result[:] = np.nan
    # First value is simple average
    result[period-1] = np.mean(source[:period])
    # Subsequent values: SMMA = (PREV_SMMA*(period-1) + PRICE) / period
    for i in range(period, len(source)):
        result[i] = (result[i-1] * (period-1) + source[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Alligator calculation (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 21:
        return np.zeros(n)
    
    # Calculate Williams Alligator on 12h
    # Jaw: 13-period SMMA of median price, shifted 8 bars
    # Teeth: 8-period SMMA of median price, shifted 5 bars  
    # Lips: 5-period SMMA of median price, shifted 3 bars
    median_price = (df_12h['high'].values + df_12h['low'].values) / 2
    
    jaw_raw = smma(median_price, 13)
    teeth_raw = smma(median_price, 8)
    lips_raw = smma(median_price, 5)
    
    # Apply shifts (shift right = add NaN at beginning)
    jaw = np.roll(jaw_raw, 8)
    jaw[:8] = np.nan
    teeth = np.roll(teeth_raw, 5)
    teeth[:5] = np.nan
    lips = np.roll(lips_raw, 3)
    lips[:3] = np.nan
    
    # Align Alligator lines to primary timeframe (12h -> 12h, so 1:1 but with shift delay)
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # Get 1d data for chop regime filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate Chopiness Index on 1d
    # CHOP = 100 * log10(sum(ATR1) / (n * log(n))) / log10(n)
    # where ATR1 = True Range, n = 14
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr1 = np.maximum(tr1, np.abs(low_1d[1:] - close_1d[:-1]))
    tr1 = np.concatenate([[np.nan], tr1])  # align with close_1d
    
    atr1 = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    atr_sum = pd.Series(atr1).rolling(window=14, min_periods=14).sum().values
    
    n_val = 14
    log_n = np.log10(n_val)
    chop = 100 * (np.log10(atr_sum) - log_n * np.log10(n_val)) / log_n
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Alligator shifts and chop
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(lips_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(jaw_aligned[i]) or
            np.isnan(chop_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        Lips = lips_aligned[i]
        Teeth = teeth_aligned[i]
        Jaw = jaw_aligned[i]
        chop_val = chop_aligned[i]
        vol_spike = volume_spike[i]
        
        # Determine trend regime from Alligator
        # Trending up: Lips > Teeth > Jaw
        # Trending down: Lips < Teeth < Jaw
        # Ranging/choppy: otherwise (lines intertwined)
        trending_up = (Lips > Teeth) and (Teeth > Jaw)
        trending_down = (Lips < Teeth) and (Teeth < Jaw)
        
        # Chop filter: only trade when 1d chop < 50 (trending) OR > 61.8 (ranging) but we use Alligator for direction
        # Actually, we want to avoid choppy markets, so require chop < 61.8 (not too choppy)
        not_choppy = chop_val < 61.8
        
        if position == 0:
            # Look for entry signals
            # Long: Alligator trending up AND volume spike AND not too choppy
            long_entry = trending_up and vol_spike and not_choppy
            # Short: Alligator trending down AND volume spike AND not too choppy
            short_entry = trending_down and vol_spike and not_choppy
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: Alligator no longer trending up OR chop too high
            if not (trending_up and not_choppy):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: Alligator no longer trending down OR chop too high
            if not (trending_down and not_choppy):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Williams_Alligator_VolumeSpike_1dChopFilter"
timeframe = "12h"
leverage = 1.0