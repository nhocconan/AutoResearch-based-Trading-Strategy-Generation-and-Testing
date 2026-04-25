#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_1dRegime_ADXFilter
Hypothesis: On 6h timeframe, use Elder Ray Bull/Bear Power (13-period EMA based) combined with 1-day ADX regime filter to capture trending moves while avoiding whipsaws in ranging markets. Bull Power > 0 and Bear Power < 0 with ADX > 25 indicates strong trend. Works in both bull and bear markets by adapting to trend strength via ADX. Designed for low trade frequency (12-37/year) to minimize fee drag.
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
    
    # 1d data for ADX regime filter (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d ADX (14-period)
    # True Range
    tr1 = df_1d['high'].values[1:] - df_1d['low'].values[1:]
    tr2 = np.abs(df_1d['high'].values[1:] - df_1d['close'].values[:-1])
    tr3 = np.abs(df_1d['low'].values[1:] - df_1d['close'].values[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    dm_plus = np.where((df_1d['high'].values[1:] - df_1d['high'].values[:-1]) > (df_1d['low'].values[:-1] - df_1d['low'].values[1:]),
                       np.maximum(df_1d['high'].values[1:] - df_1d['high'].values[:-1], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.where((df_1d['low'].values[:-1] - df_1d['low'].values[1:]) > (df_1d['high'].values[1:] - df_1d['high'].values[:-1]),
                        np.maximum(df_1d['low'].values[:-1] - df_1d['low'].values[1:], 0), 0)
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed DM and TR
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    tr_smooth = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / tr_smooth
    di_minus = 100 * dm_minus_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align 1d ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 6h Elder Ray Bull/Bear Power (13-period EMA)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13  # Bull Power = High - EMA13
    bear_power = low - ema_13   # Bear Power = Low - EMA13
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for 1d ADX (14+14+14=42 for smoothing) and 6h EMA (13)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema_13[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: ADX > 25 indicates trending market
        trending = adx_aligned[i] > 25
        
        if position == 0:
            # Look for entry signals
            # Long: Bull Power > 0 (strong buying pressure) in trending market
            long_entry = (bull_power[i] > 0) and trending and volume_spike[i]
            # Short: Bear Power < 0 (strong selling pressure) in trending market
            short_entry = (bear_power[i] < 0) and trending and volume_spike[i]
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit when Bull Power turns negative or ADX weakens
            if bull_power[i] <= 0 or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when Bear Power turns positive or ADX weakens
            if bear_power[i] >= 0 or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_BullBearPower_1dRegime_ADXFilter"
timeframe = "6h"
leverage = 1.0