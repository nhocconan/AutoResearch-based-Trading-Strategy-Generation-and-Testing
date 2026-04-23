#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator + Elder Ray with 1d volume spike and chop regime filter.
- Williams Alligator: Jaw (13), Teeth (8), Lips (5) SMAs with 8,5,3 offsets
- Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
- Long: Alligator aligned bullish (Lips>Teeth>Jaw) + Bull Power > 0 + volume > 1.5x 20-period avg + CHOP > 61.8 (range)
- Short: Alligator aligned bearish (Lips<Teeth<Jaw) + Bear Power < 0 + volume > 1.5x 20-period avg + CHOP > 61.8 (range)
- Exit: Opposite Alligator alignment or volume < 1.2x average
- Uses 1d timeframe for HTF indicators to reduce noise, 12h for entries
- CHOP filter ensures we only trade in ranging markets where Alligator/Elder Ray work best
- Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe
- Discrete position sizing: ±0.25 to minimize fee churn
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d Williams Alligator
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Alligator SMAs
    jaw_1d = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().values  # Jaw (13)
    teeth_1d = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().values   # Teeth (8)
    lips_1d = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().values    # Lips (5)
    
    # Apply Alligator offsets (shift forward)
    jaw_1d = np.roll(jaw_1d, 8)
    teeth_1d = np.roll(teeth_1d, 5)
    lips_1d = np.roll(lips_1d, 3)
    
    # Align to 12h timeframe
    jaw_1d_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_1d_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_1d_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    
    # Calculate 1d Elder Ray (requires EMA13)
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power_1d = high_1d - ema_13_1d  # High - EMA13
    bear_power_1d = low_1d - ema_13_1d   # Low - EMA13
    
    # Align Elder Ray to 12h
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Calculate 1d Choppiness Index (CHOP) for regime filter
    # CHOP = 100 * log10(sum(ATR(14)) / (n * (HHV - LLV))) / log10(n)
    # We'll use a simplified version: high-low range over period vs true range
    atr_period = 14
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(np.maximum(tr1, tr2), tr3)
    atr_1d = pd.Series(tr_1d).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    hh_1d = pd.Series(high_1d).rolling(window=atr_period, min_periods=atr_period).max().values
    ll_1d = pd.Series(low_1d).rolling(window=atr_period, min_periods=atr_period).min().values
    
    # Avoid division by zero
    range_1d = hh_1d - ll_1d
    range_1d = np.where(range_1d == 0, 1e-10, range_1d)
    
    chop_1d = 100 * (np.log10(pd.Series(atr_1d).rolling(window=atr_period, min_periods=atr_period).sum().values) 
                      - np.log10(atr_period * range_1d)) / np.log10(atr_period)
    
    # Align CHOP to 12h
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 30)  # Need enough for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(jaw_1d_aligned[i]) or
            np.isnan(teeth_1d_aligned[i]) or
            np.isnan(lips_1d_aligned[i]) or
            np.isnan(bull_power_1d_aligned[i]) or
            np.isnan(bear_power_1d_aligned[i]) or
            np.isnan(chop_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Chop regime filter: CHOP > 61.8 indicates ranging market (good for Alligator/Elder Ray)
        chop_filter = chop_1d_aligned[i] > 61.8
        
        if position == 0:
            # Long: Alligator bullish + Bull Power > 0 + volume confirmation + chop regime
            if (lips_1d_aligned[i] > teeth_1d_aligned[i] > jaw_1d_aligned[i] and
                bull_power_1d_aligned[i] > 0 and
                volume_confirm and
                chop_filter):
                signals[i] = 0.25
                position = 1
            # Short: Alligator bearish + Bear Power < 0 + volume confirmation + chop regime
            elif (lips_1d_aligned[i] < teeth_1d_aligned[i] < jaw_1d_aligned[i] and
                  bear_power_1d_aligned[i] < 0 and
                  volume_confirm and
                  chop_filter):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Alligator alignment breaks OR volume drops
            if not (lips_1d_aligned[i] > teeth_1d_aligned[i] > jaw_1d_aligned[i]) or not volume_confirm:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator alignment breaks OR volume drops
            if not (lips_1d_aligned[i] < teeth_1d_aligned[i] < jaw_1d_aligned[i]) or not volume_confirm:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_ElderRay_1dVolumeChop"
timeframe = "12h"
leverage = 1.0