# Solution
#!/usr/bin/env python3
"""
Hypothesis: 4-hour Donchian breakout with 1-day volatility filter and volume confirmation.
Long when price breaks above Donchian upper band, ATR ratio > 1.5, and volume > 1.5x SMA volume.
Short when price breaks below Donchian lower band, ATR ratio > 1.5, and volume > 1.5x SMA volume.
Exit when price crosses opposite Donchian band or ATR ratio falls below 0.8.
Donchian channels provide clear breakout levels; volatility filter ensures breakouts occur during
expanding volatility; volume confirmation adds conviction. Designed for low trade frequency by requiring
multiple confirmations and using 4h timeframe. Works in both bull and bear markets by capturing
breakouts in either direction with strict entry conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1-day data for ATR calculation - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-day ATR
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    atr_14 = np.zeros_like(tr)
    atr_14[:14] = np.nan
    for i in range(14, len(tr)):
        atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Donchian channels (20 periods) on 4h data
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_20
    donchian_lower = low_20
    
    # Volume confirmation: volume > 1.5x 20-period SMA volume
    volume_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(volume_sma[i]) or np.isnan(atr_14_aligned[i]) or atr_14_aligned[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate current ATR ratio (current volatility vs 1-day ATR)
        # Use 4-period high-low range as proxy for current volatility
        if i >= 3:
            curr_range = np.max(high[i-3:i+1]) - np.min(low[i-3:i+1])
            atr_ratio = curr_range / atr_14_aligned[i] if atr_14_aligned[i] > 0 else 0
        else:
            atr_ratio = 0
        
        if position == 0:
            # Long: Price breaks above upper band, ATR ratio > 1.5, volume > 1.5x SMA
            if (close[i] > donchian_upper[i] and 
                atr_ratio > 1.5 and 
                volume[i] > 1.5 * volume_sma[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower band, ATR ratio > 1.5, volume > 1.5x SMA
            elif (close[i] < donchian_lower[i] and 
                  atr_ratio > 1.5 and 
                  volume[i] > 1.5 * volume_sma[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price falls below lower band OR ATR ratio falls below 0.8
                if (close[i] < donchian_lower[i] or 
                    atr_ratio < 0.8):
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price rises above upper band OR ATR ratio falls below 0.8
                if (close[i] > donchian_upper[i] or 
                    atr_ratio < 0.8):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian_Breakout_1dATR_Volume"
timeframe = "4h"
leverage = 1.0