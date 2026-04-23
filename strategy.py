#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ATR regime filter and volume confirmation.
Long when price breaks above 4h Donchian upper band AND 1d ATR(14) < 30-day median ATR (low volatility regime) AND volume > 1.5x average.
Short when price breaks below 4h Donchian lower band AND 1d ATR(14) < 30-day median ATR AND volume > 1.5x average.
Exit when price reverts to 4h Donchian middle band (20-period average of high/low).
Uses 4h timeframe with volatility regime filter to avoid whipsaws in high volatility periods.
Target: 100-180 trades over 4 years (25-45/year) to stay within proven working range.
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
    
    # Calculate 4h Donchian channels (20-period) - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Donchian channels on 4h (based on previous 20 periods)
    def rolling_max(arr, window):
        return np.convolve(arr, np.ones(window), 'valid')[:len(arr)-window+1] if len(arr) >= window else np.full(len(arr), np.nan)
    def rolling_min(arr, window):
        return np.convolve(arr, np.ones(window), 'valid')[:len(arr)-window+1] if len(arr) >= window else np.full(len(arr), np.nan)
    
    # Simpler approach using pandas rolling
    high_max_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_upper = np.full(len(high_4h), np.nan)
    donchian_lower = np.full(len(high_4h), np.nan)
    donchian_middle = np.full(len(high_4h), np.nan)
    
    if len(high_max_20) >= 20:
        donchian_upper[19:] = high_max_20[:len(high_max_20)]
        donchian_lower[19:] = low_min_20[:len(low_min_20)]
        donchian_middle[19:] = (high_max_20[:len(high_max_20)] + low_min_20[:len(low_min_20)]) / 2.0
    
    # Load 1d data for ATR regime filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # ATR(14)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 30-day median ATR for regime filter
    atr_median_30 = pd.Series(atr_14).rolling(window=30, min_periods=30).median().values
    
    # Align HTF indicators to 4h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_4h, donchian_middle)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    atr_median_30_aligned = align_htf_to_ltf(prices, df_1d, atr_median_30)
    
    # Volume average (20-period) on primary timeframe
    vol_ma_primary = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or np.isnan(donchian_middle_aligned[i]) or 
            np.isnan(atr_14_aligned[i]) or np.isnan(atr_median_30_aligned[i]) or np.isnan(vol_ma_primary[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        upper_val = donchian_upper_aligned[i]
        lower_val = donchian_lower_aligned[i]
        middle_val = donchian_middle_aligned[i]
        atr_val = atr_14_aligned[i]
        atr_med_val = atr_median_30_aligned[i]
        vol_ma_val = vol_ma_primary[i]
        
        # Get current price and volume
        price = close[i]
        vol_current = volume[i]
        
        # Regime filter: only trade in low volatility (ATR < median ATR)
        low_volatility_regime = atr_val < atr_med_val
        
        if position == 0:
            # Long: price breaks above Donchian upper AND low volatility regime AND volume spike
            if price > upper_val and low_volatility_regime and vol_current > 1.5 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower AND low volatility regime AND volume spike
            elif price < lower_val and low_volatility_regime and vol_current > 1.5 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price reverts to Donchian middle band
                if price <= middle_val:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price reverts to Donchian middle band
                if price >= middle_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian20_ATRRegime_VolumeBreakout"
timeframe = "4h"
leverage = 1.0