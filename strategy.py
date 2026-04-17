#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with volume spike and 1d ATR-based volatility filter.
Long when price breaks above Donchian upper band AND volume > 2.0x 20-period average AND 1d ATR ratio (current/20MA) > 1.2 (elevated volatility).
Short when price breaks below Donchian lower band AND volume > 2.0x average AND 1d ATR ratio > 1.2.
Exit when price reverts to Donchian middle OR ATR ratio < 0.8 (low volatility).
Uses volume spike for confirmation of institutional interest and ATR regime filter to trade only in elevated volatility environments.
Target: 75-200 total trades over 4 years (19-50/year). Works in bull markets (breaks highs with volume) and bear markets (breaks lows with panic volume).
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
    
    # Get 4h data for Donchian calculation
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate Donchian channels on 4h timeframe (20-period)
    high_4h_series = pd.Series(high_4h)
    low_4h_series = pd.Series(low_4h)
    donchian_upper = high_4h_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_4h_series.rolling(window=20, min_periods=20).min().values
    donchian_middle = ((donchian_upper + donchian_lower) / 2).values
    
    # Volume average (20-period) on 4h
    volume_ma = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for ATR-based volatility filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range and ATR on 1d timeframe
    high_1d_series = pd.Series(high_1d)
    low_1d_series = pd.Series(low_1d)
    close_1d_series = pd.Series(close_1d)
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    # ATR (14-period)
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    # ATR 20-period moving average for regime filter
    atr_ma_1d = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    # ATR ratio: current ATR / ATR MA (values >1 = elevated volatility)
    atr_ratio = np.where(atr_ma_1d != 0, atr_1d / atr_ma_1d, 1.0)
    
    # Align 4h indicators to 4h timeframe (no alignment needed for same timeframe)
    donchian_upper_aligned = donchian_upper
    donchian_lower_aligned = donchian_lower
    donchian_middle_aligned = donchian_middle
    volume_ma_aligned = align_htf_to_ltf(prices, df_4h, volume_ma)
    
    # Align 1d ATR ratio to 4h timeframe
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 60  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_middle_aligned[i]) or np.isnan(volume_ma_aligned[i]) or 
            np.isnan(atr_ratio_aligned[i])):
            signals[i] = 0.0
            continue
        
        du = donchian_upper_aligned[i]
        dl = donchian_lower_aligned[i]
        dm = donchian_middle_aligned[i]
        vol_ma = volume_ma_aligned[i]
        vol = volume[i]
        atr_ratio_val = atr_ratio_aligned[i]
        price = close[i]
        
        if position == 0:
            # Long: price > Donchian upper AND volume > 2.0x avg AND ATR ratio > 1.2 (elevated vol)
            if price > du and vol > 2.0 * vol_ma and atr_ratio_val > 1.2:
                signals[i] = 0.25
                position = 1
            # Short: price < Donchian lower AND volume > 2.0x avg AND ATR ratio > 1.2
            elif price < dl and vol > 2.0 * vol_ma and atr_ratio_val > 1.2:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price < Donchian middle OR ATR ratio < 0.8 (low volatility)
            if price < dm or atr_ratio_val < 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price > Donchian middle OR ATR ratio < 0.8 (low volatility)
            if price > dm or atr_ratio_val < 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_VolumeSpike_ATRRegime"
timeframe = "4h"
leverage = 1.0