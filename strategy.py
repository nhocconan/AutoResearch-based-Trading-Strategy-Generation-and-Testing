#!/usr/bin/env python3

"""
Hypothesis: 12-hour Donchian Channel Breakout with 1-day ATR volatility filter and volume confirmation.
Trades breakouts of the 20-period Donchian channel (upper/lower bands) in the direction of the breakout,
filtered by elevated ATR (volatility expansion) and volume spike to avoid false breakouts in low volatility.
Designed for low trade frequency (15-30 trades/year) to minimize fee drag and work in both bull and bear
markets by requiring volatility expansion and volume confirmation, reducing whipsaws during ranging periods.
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
    
    # Load daily data for ATR and trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Daily ATR (14-period) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # 12-hour Donchian channel (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike: current volume > 1.5x 30-period average
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(atr_14_1d_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ma_30[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility filter: ATR above 50-period average (expanding volatility)
        atr_ma_50 = pd.Series(atr_14_1d_aligned).rolling(window=50, min_periods=50).mean().values
        vol_expansion = atr_14_1d_aligned[i] > atr_ma_50[i]
        
        # Volume confirmation
        vol_spike = volume[i] > 1.5 * vol_ma_30[i]
        
        if position == 0 and vol_expansion and vol_spike:
            # Long: breakout above Donchian high
            if high[i] > donchian_high[i]:
                signals[i] = 0.25
                position = 1
            # Short: breakout below Donchian low
            elif low[i] < donchian_low[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: opposite Donchian breakout or volatility contraction
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below Donchian low
                if low[i] < donchian_low[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price breaks above Donchian high
                if high[i] > donchian_high[i]:
                    exit_signal = True
            
            # Also exit if volatility contracts significantly
            if atr_14_1d_aligned[i] < 0.7 * atr_ma_50[i]:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Donchian20_ATR14_Volume30_Breakout"
timeframe = "12h"
leverage = 1.0