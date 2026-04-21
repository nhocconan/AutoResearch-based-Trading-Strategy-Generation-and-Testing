#!/usr/bin/env python3
"""
6h_ElderRay_RegimeFilter_V1
Hypothesis: Use 6h Elder Ray (Bull/Bear Power) with 1d ADX regime filter. 
- Bull Power = High - EMA13, Bear Power = EMA13 - Low
- Enter long when Bull Power > 0 and rising + ADX > 25 (trending)
- Enter short when Bear Power > 0 and rising + ADX > 25 (trending)
- Exit when power falls below zero or ADX < 20 (range)
- Uses 1d EMA13 and ADX for multi-timeframe alignment to reduce whipsaw.
- Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')  # for EMA13 and ADX
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d EMA13 for Elder Ray Power ===
    close_1d = df_1d['close'].values
    ema13 = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema13_aligned = align_htf_to_ltf(prices, df_1d, ema13)
    
    # === 1d ADX for Regime Filter ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = dm_minus[0] = 0
    
    # Smoothed DM and TR
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    tr_smooth = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / (tr_smooth + 1e-10)
    di_minus = 100 * dm_minus_smooth / (tr_smooth + 1e-10)
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # === 6h Indicators ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Elder Ray Power: Bull Power = High - EMA13, Bear Power = EMA13 - Low
    bull_power = high - ema13_aligned
    bear_power = ema13_aligned - low
    
    # Rising power condition (current > previous)
    bull_power_rising = bull_power > np.roll(bull_power, 1)
    bear_power_rising = bear_power > np.roll(bear_power, 1)
    bull_power_rising[0] = False
    bear_power_rising[0] = False
    
    # Volume filter (optional confirmation)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ok = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema13_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime: ADX > 25 = trending, ADX < 20 = range (hysteresis)
        is_trending = adx_aligned[i] > 25
        is_range = adx_aligned[i] < 20
        
        if position == 0:
            # Enter long: Bull Power > 0 and rising + trending regime
            if bull_power[i] > 0 and bull_power_rising[i] and is_trending and vol_ok[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: Bear Power > 0 and rising + trending regime
            elif bear_power[i] > 0 and bear_power_rising[i] and is_trending and vol_ok[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Bull Power <= 0 or regime turns range
            if bull_power[i] <= 0 or is_range:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Bear Power <= 0 or regime turns range
            if bear_power[i] <= 0 or is_range:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_RegimeFilter_V1"
timeframe = "6h"
leverage = 1.0