#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Index with 12h ADX trend filter and volume confirmation.
Long when Bull Power > 0 and Bear Power < 0 (bullish momentum), 12h ADX > 20 (trending market),
and volume > 1.2x average. Short when Bear Power > 0 and Bull Power < 0 (bearish momentum),
12h ADX > 20, and volume > 1.2x average. Exit when momentum diverges (Bull Power <= 0 for long,
Bear Power <= 0 for short) or trend weakens (ADX < 15). Designed for low trade frequency
(~15-30/year) to capture momentum moves in trending markets while avoiding whipsaws in ranges.
Works in both bull and bear markets by requiring trend confirmation (ADX > 20) for momentum entries.
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
    
    # Load 12h data for EMA13 (Elder Ray) and ADX - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate EMA13 for 12h (Elder Ray base)
    ema13 = pd.Series(close_12h).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high_12h - ema13
    bear_power = low_12h - ema13
    
    # Calculate 12h ADX (14-period)
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    dm_plus = np.where((high_12h - np.roll(high_12h, 1)) > (np.roll(low_12h, 1) - low_12h),
                       np.maximum(high_12h - np.roll(high_12h, 1), 0), 0)
    dm_minus = np.where((np.roll(low_12h, 1) - low_12h) > (high_12h - np.roll(high_12h, 1)),
                        np.maximum(np.roll(low_12h, 1) - low_12h, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum()
    dm_plus14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum()
    dm_minus14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum()
    
    plus_di = 100 * dm_plus14 / tr14
    minus_di = 100 * dm_minus14 / tr14
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean()
    adx_values = adx.values
    
    # Align HTF indicators to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_12h, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_12h, bear_power)
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx_values)
    
    # Volume average (20-period) on 6h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        bull_val = bull_power_aligned[i]
        bear_val = bear_power_aligned[i]
        adx_val = adx_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: Bull Power > 0 and Bear Power < 0 (bullish momentum), ADX > 20 (trend), volume confirmation
            if (bull_val > 0 and bear_val < 0 and
                adx_val > 20 and vol_current > 1.2 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power > 0 and Bull Power < 0 (bearish momentum), ADX > 20, volume confirmation
            elif (bear_val > 0 and bull_val < 0 and
                  adx_val > 20 and vol_current > 1.2 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Bull Power <= 0 (momentum fading) OR ADX < 15 (trend weakening)
                if bull_val <= 0 or adx_val < 15:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Bear Power <= 0 (momentum fading) OR ADX < 15
                if bear_val <= 0 or adx_val < 15:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_ElderRay_12hADX_Volume_Momentum"
timeframe = "6h"
leverage = 1.0