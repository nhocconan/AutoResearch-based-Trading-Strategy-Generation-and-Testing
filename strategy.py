#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d ATR volatility filter and volume confirmation.
Long when price breaks above Donchian upper band (20-period high) and ATR(14) > 1d ATR(14) median with volume > 1.5x average.
Short when price breaks below Donchian lower band (20-period low) and ATR(14) > 1d ATR(14) median with volume > 1.5x average.
Exit on opposite Donchian band break. Uses 12h timeframe targeting 50-150 total trades over 4 years.
Donchian channels provide clear breakout levels, ATR filter ensures sufficient volatility, volume confirms strength.
Designed to capture strong momentum moves while avoiding low-volatility whipsaws across both bull and bear regimes.
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
    
    # Load 1d data for ATR filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR(14)
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    # Prepend NaN for first element to align with close_1d indexing
    atr_1d = np.concatenate([[np.nan], atr_1d])
    # Calculate median of 1d ATR for filter
    atr_1d_median = np.nanmedian(atr_1d[-100:]) if len(atr_1d) >= 100 else np.nanmedian(atr_1d[~np.isnan(atr_1d)])
    
    # Align 1d ATR median to 12h timeframe (constant value)
    atr_1d_median_aligned = np.full(n, atr_1d_median)
    
    # Donchian(20) on 12h timeframe
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # ATR(14) on 12h timeframe for volatility filter
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = np.nan  # First element has no previous close
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or np.isnan(atr[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr_1d_median_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        donchian_upper = high_roll[i]
        donchian_lower = low_roll[i]
        atr_12h = atr[i]
        atr_1d_med = atr_1d_median_aligned[i]
        vol_ma_val = vol_ma[i]
        price = close[i]
        vol_current = volume[i]
        
        # Volatility filter: current 12h ATR > 1d ATR median
        vol_filter = atr_12h > atr_1d_med
        # Volume confirmation: current volume > 1.5x average
        vol_confirm = vol_current > 1.5 * vol_ma_val
        
        if position == 0:
            # Long: price breaks above Donchian upper band with volatility and volume confirmation
            if (price > donchian_upper and vol_filter and vol_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower band with volatility and volume confirmation
            elif (price < donchian_lower and vol_filter and vol_confirm):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below Donchian lower band
                if price < donchian_lower:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price breaks above Donchian upper band
                if price > donchian_upper:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Donchian20_1dATR_VolumeSpike"
timeframe = "12h"
leverage = 1.0