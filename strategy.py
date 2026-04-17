#!/usr/bin/env python3
"""
Hypothesis: 4h timeframe with 12h ATR-based volatility expansion filter + Donchian(20) breakout + volume confirmation.
Long when price breaks above 4h Donchian(20) high with 12h ATR(7)/ATR(30) > 1.8 (expanding volatility) and volume > 1.5x 20-period volume average.
Short when price breaks below 4h Donchian(20) low with 12h ATR(7)/ATR(30) > 1.8 and volume > 1.5x 20-period volume average.
Volatility expansion captures momentum after consolidation, effective in both bull and bear markets.
Uses discrete position sizing (0.25) to minimize fee churn and targeting ~30 trades/year.
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
    
    # Get 12h data for ATR-based volatility expansion
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h ATR(7) and ATR(30)
    def atr(high_vals, low_vals, close_vals, window):
        tr1 = high_vals - low_vals
        tr2 = np.abs(high_vals - np.roll(close_vals, 1))
        tr3 = np.abs(low_vals - np.roll(close_vals, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # first period TR is high-low
        atr_vals = pd.Series(tr).rolling(window=window, min_periods=window).mean().values
        return atr_vals
    
    atr_7_12h = atr(high_12h, low_12h, close_12h, 7)
    atr_30_12h = atr(high_12h, low_12h, close_12h, 30)
    
    # Avoid division by zero
    atr_ratio_12h = np.where(atr_30_12h != 0, atr_7_12h / atr_30_12h, 1.0)
    
    # Calculate 4h Donchian(20) channels
    def donchian_channel(high_vals, low_vals, window):
        upper = pd.Series(high_vals).rolling(window=window, min_periods=window).max().values
        lower = pd.Series(low_vals).rolling(window=window, min_periods=window).min().values
        return upper, lower
    
    donchian_upper, donchian_lower = donchian_channel(high, low, 20)
    
    # Calculate 4h volume 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 12h ATR ratio to 4h timeframe
    atr_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_ratio_12h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 30  # need enough for ATR(30) and Donchian
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(atr_ratio_12h_aligned[i]) or 
            np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volatility expansion filter: ATR(7)/ATR(30) > 1.8
        vol_expansion = atr_ratio_12h_aligned[i] > 1.8
        
        # Volume confirmation: current 4h volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
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

name = "4h_12hATRratio_VolExpansion_Donchian20_Breakout_Volume_Confirm"
timeframe = "4h"
leverage = 1.0