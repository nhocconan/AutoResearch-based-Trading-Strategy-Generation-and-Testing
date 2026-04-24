#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with 1w Elder Ray trend filter and volume spike confirmation.
- Williams Alligator: Jaw (13), Teeth (8), Lips (5) SMAs smoothed by 8,5,3 periods respectively.
  Long when Lips > Teeth > Jaw (bullish alignment), Short when Lips < Teeth < Jaw (bearish alignment).
- 1w Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low.
  Long filter: Bull Power > 0 and rising, Short filter: Bear Power > 0 and rising.
- Volume confirmation: Current volume > 2.0 * median volume of last 28 bars (avoid low-volume noise).
- Entry: Alligator alignment + Elder Ray filter + volume spike.
- Exit: Opposite Alligator alignment or Elder Ray filter failure.
- Uses 12h primary timeframe with 1w HTF to target 50-150 total trades over 4 years (12-37/year).
- Williams Alligator identifies trend initiation and continuation with built-in smoothing.
- 1w Elder Ray ensures alignment with weekly momentum to avoid counter-trend entries.
- Volume spike filters out low-conviction moves, reducing false signals in choppy markets.
- Designed for BTC/ETH with edge in both trending (Alligator alignment) and momentum (Elder Ray) markets.
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
    
    # Calculate Williams Alligator on 12h timeframe
    jaw_period = 13
    teeth_period = 8
    lips_period = 5
    jaw_smooth = 8
    teeth_smooth = 5
    lips_smooth = 3
    
    # Jaw: 13-period SMA smoothed by 8 periods
    sma_jaw = pd.Series(close).rolling(window=jaw_period, min_periods=jaw_period).mean().values
    jaw = pd.Series(sma_jaw).rolling(window=jaw_smooth, min_periods=jaw_smooth).mean().values
    
    # Teeth: 8-period SMA smoothed by 5 periods
    sma_teeth = pd.Series(close).rolling(window=teeth_period, min_periods=teeth_period).mean().values
    teeth = pd.Series(sma_teeth).rolling(window=teeth_smooth, min_periods=teeth_smooth).mean().values
    
    # Lips: 5-period SMA smoothed by 3 periods
    sma_lips = pd.Series(close).rolling(window=lips_period, min_periods=lips_period).mean().values
    lips = pd.Series(sma_lips).rolling(window=lips_smooth, min_periods=lips_smooth).mean().values
    
    # Get 1w data ONCE before loop for Elder Ray trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 13:
        return np.zeros(n)
    
    # Calculate 1w EMA13 for Elder Ray
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    ema_13_1w = pd.Series(close_1w).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
    bull_power = high_1w - ema_13_1w
    bear_power = ema_13_1w - low_1w
    
    # Align 1w indicators to 12h timeframe
    ema_13_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_13_1w)
    bull_power_aligned = align_htf_to_ltf(prices, df_1w, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1w, bear_power)
    
    # Volume confirmation: volume > 2.0 * median volume of last 28 bars
    vol_median = pd.Series(volume).rolling(window=28, min_periods=28).median().values
    volume_confirm = volume > (2.0 * vol_median)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(jaw_period + jaw_smooth, teeth_period + teeth_smooth, lips_period + lips_smooth, 28) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_13_1w_aligned[i]) or np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or np.isnan(vol_median[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Alligator alignment conditions
        lips_above_teeth = lips[i] > teeth[i]
        teeth_above_jaw = teeth[i] > jaw[i]
        lips_below_teeth = lips[i] < teeth[i]
        teeth_below_jaw = teeth[i] < jaw[i]
        
        bullish_alignment = lips_above_teeth and teeth_above_jaw
        bearish_alignment = lips_below_teeth and teeth_below_jaw
        
        # Elder Ray conditions: power > 0 and rising (current > previous)
        bull_power_rising = bull_power_aligned[i] > 0 and (i == start_idx or bull_power_aligned[i] > bull_power_aligned[i-1])
        bear_power_rising = bear_power_aligned[i] > 0 and (i == start_idx or bear_power_aligned[i] > bear_power_aligned[i-1])
        
        if position == 0:
            # Long: bullish Alligator alignment + bull Elder Ray rising + volume confirmation
            if bullish_alignment and bull_power_rising and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish Alligator alignment + bear Elder Ray rising + volume confirmation
            elif bearish_alignment and bear_power_rising and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: bearish Alligator alignment OR bull Elder Ray not rising
            if not bullish_alignment or not bull_power_rising:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: bullish Alligator alignment OR bear Elder Ray not rising
            if not bearish_alignment or not bear_power_rising:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Williams_Alligator_1w_ElderRay_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0