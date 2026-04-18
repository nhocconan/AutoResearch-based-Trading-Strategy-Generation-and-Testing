#!/usr/bin/env python3
"""
1d_Williams_Alligator_ElderRay_Volume
Hypothesis: Daily Williams Alligator trend filter combined with Elder Ray power and volume spikes.
Alligator (jaw/teeth/lips) defines trend direction, Elder Ray measures bull/bear power,
volume confirms strength. Designed for low trade frequency (15-25/year) with robustness
in both bull and bear markets via trend-following with momentum confirmation.
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
    
    # Get weekly data for higher timeframe trend filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Williams Alligator on weekly: SMAs with specific offsets
    # Jaw: 13-period SMA, shifted 8 bars forward
    # Teeth: 8-period SMA, shifted 5 bars forward  
    # Lips: 5-period SMA, shifted 3 bars forward
    def sma_with_shift(arr, period, shift):
        sma = np.full(len(arr), np.nan)
        if len(arr) >= period:
            for i in range(period - 1, len(arr)):
                sma[i] = np.mean(arr[i - period + 1:i + 1])
        # Apply shift forward (look-ahead prevented by alignment later)
        shifted = np.full(len(arr), np.nan)
        if len(sma) > shift:
            shifted[shift:] = sma[:-shift]
        return shifted
    
    jaw = sma_with_shift(close_1w, 13, 8)
    teeth = sma_with_shift(close_1w, 8, 5)
    lips = sma_with_shift(close_1w, 5, 3)
    
    # Align Alligator components to daily timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1w, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1w, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1w, lips)
    
    # Elder Ray Power on daily: Bull Power = High - EMA13, Bear Power = Low - EMA13
    def calculate_ema(arr, period):
        ema = np.full(len(arr), np.nan)
        if len(arr) >= period:
            k = 2 / (period + 1)
            ema[period - 1] = np.mean(arr[0:period])
            for i in range(period, len(arr)):
                ema[i] = arr[i] * k + ema[i - 1] * (1 - k)
        return ema
    
    ema13_daily = calculate_ema(close, 13)
    bull_power = high - ema13_daily
    bear_power = low - ema13_daily
    
    # Volume spike: current volume > 2.5 x 20-day average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 2.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 13)  # Ensure indicators ready
    
    for i in range(start_idx, n):
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Alligator alignment: Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend
        is_uptrend = lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i]
        is_downtrend = lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i]
        
        if position == 0:
            # Long: Alligator uptrend + positive Bull Power + volume spike
            if is_uptrend and bull_power[i] > 0 and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Alligator downtrend + negative Bear Power + volume spike
            elif is_downtrend and bear_power[i] < 0 and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Alligator turns down OR Bull Power turns negative
            if not is_uptrend or bull_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Alligator turns up OR Bear Power turns positive
            if not is_downtrend or bear_power[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Williams_Alligator_ElderRay_Volume"
timeframe = "1d"
leverage = 1.0