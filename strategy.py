#!/usr/bin/env python3
"""
Hypothesis: 4h timeframe with 1d ATR-based volatility expansion filter + Donchian(20) breakout + volume confirmation.
Long when price breaks above 4h Donchian(20) high with 1d ATR(14)/ATR(50) > 1.2 (expanding volatility) and volume > 1.3x 20-period volume average.
Short when price breaks below 4h Donchian(20) low with 1d ATR(14)/ATR(50) > 1.2 and volume > 1.3x 20-period volume average.
Volatility expansion indicates increased participation and higher probability of sustained breakout.
Designed to work in bull markets (breakouts with rising volatility) and bear markets (breakdowns with rising volatility).
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
    
    # Get 1d data for ATR calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR(14) and ATR(50)
    def atr(high_vals, low_vals, close_vals, window):
        tr1 = high_vals - low_vals
        tr2 = np.abs(high_vals - np.concatenate([[close_vals[0]], close_vals[:-1]]))
        tr3 = np.abs(low_vals - np.concatenate([[close_vals[0]], close_vals[:-1]]))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr_vals = pd.Series(tr).ewm(span=window, adjust=False, min_periods=window).mean().values
        return atr_vals
    
    atr_14_1d = atr(high_1d, low_1d, close_1d, 14)
    atr_50_1d = atr(high_1d, low_1d, close_1d, 50)
    
    # Calculate 4h Donchian(20) channels
    def donchian_channel(high_vals, low_vals, window):
        upper = pd.Series(high_vals).rolling(window=window, min_periods=window).max().values
        lower = pd.Series(low_vals).rolling(window=window, min_periods=window).min().values
        return upper, lower
    
    donchian_upper, donchian_lower = donchian_channel(high, low, 20)
    
    # Calculate 4h volume 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d ATR values to 4h timeframe
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    atr_50_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_50_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # need enough for ATR(50) and Donchian/volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(atr_50_1d_aligned[i]) or 
            np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volatility expansion filter: ATR(14) > 1.2 * ATR(50)
        vol_expansion = atr_14_1d_aligned[i] > 1.2 * atr_50_1d_aligned[i]
        
        # Volume confirmation: current 4h volume > 1.3x 20-period average
        volume_confirmed = volume[i] > 1.3 * vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above 4h Donchian(20) high with vol expansion and volume
            if (close[i] > donchian_upper[i] and 
                vol_expansion and 
                volume_confirmed):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 4h Donchian(20) low with vol expansion and volume
            elif (close[i] < donchian_lower[i] and 
                  vol_expansion and 
                  volume_confirmed):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below 4h Donchian(20) low (opposite side of channel)
            if close[i] < donchian_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above 4h Donchian(20) high (opposite side of channel)
            if close[i] > donchian_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1dATR_VolatilityExpansion_Donchian20_Breakout_Volume_Confirm"
timeframe = "4h"
leverage = 1.0