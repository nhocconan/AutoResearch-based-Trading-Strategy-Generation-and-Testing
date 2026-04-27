#!/usr/bin/env python3
"""
4h_Camarilla_Pivot_VolumeSpike_ChopRegime
Hypothesis: 4h strategy using Camarilla R3/S3 levels from 1d for breakout entries with volume confirmation and choppiness regime filter. 
Enters long when price closes above R3 with volume > 2.0x 20-period average and CHOP > 61.8 (ranging market). 
Enters short when price closes below S3 with volume confirmation and CHOP > 61.8. 
Exits on opposite Camarilla level touch (S3/R3). 
Designed for low trade frequency (<25/year) with discrete position sizing to minimize fee drag.
Uses choppiness index to avoid trending markets where Camarilla levels fail, focusing on ranging conditions for mean reversion.
Works in both bull and bear markets by exploiting mean reversion in ranging regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d OHLC for Camarilla levels
    o_1d = df_1d['open'].values
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    # Camarilla levels: R3/S3 from 1d OHLC (wider than R1/S1 for fewer false breakouts)
    # R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    camarilla_r3 = c_1d + (h_1d - l_1d) * 1.1 / 4
    camarilla_s3 = c_1d - (h_1d - l_1d) * 1.1 / 4
    
    # Align 1d indicators to 4h timeframe (completed bars only)
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    # Choppiness Index regime filter: CHOP > 61.8 = ranging market (good for mean reversion)
    # CHOP = 100 * log10(sum(ATR(14)) / (log10(n) * (highest_high - lowest_low))) 
    # Simplified: CHOP > 61.8 indicates ranging, < 38.2 indicates trending
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]  # First bar
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_atr14 = pd.Series(atr14).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range_hl = highest_high - lowest_low
    # Avoid division by zero
    chop_raw = 100 * np.log10(sum_atr14 / (np.log10(14) * range_hl + 1e-10))
    chop_raw = np.where(range_hl == 0, 50.0, chop_raw)  # Neutral when no range
    chop_raw = np.where(np.isnan(chop_raw), 50.0, chop_raw)
    chop_regime = chop_raw > 61.8  # Ranging market
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital (discrete level)
    
    # Warmup: need 1d (no lookback) + ATR14 (14) + volume avg (20)
    start_idx = max(14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(volume_confirm[i]) or np.isnan(chop_regime[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        vol_conf = volume_confirm[i]
        chop_reg = chop_regime[i]
        
        if position == 0:
            # Look for entry: Camarilla R3/S3 breakout with volume confirmation and chop regime (ranging)
            # Long: price closes above R3 AND volume confirmation AND chop > 61.8 (ranging)
            long_condition = (close_val > r3_val) and vol_conf and chop_reg
            # Short: price closes below S3 AND volume confirmation AND chop > 61.8 (ranging)
            short_condition = (close_val < s3_val) and vol_conf and chop_reg
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: price touches S3 (opposite level)
            if close_val < s3_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price touches R3 (opposite level)
            if close_val > r3_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_Pivot_VolumeSpike_ChopRegime"
timeframe = "4h"
leverage = 1.0