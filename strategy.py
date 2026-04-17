#!/usr/bin/env python3
"""
Hypothesis: 12h timeframe with 1d Donchian(20) breakout + volume confirmation + ATR trailing stop.
Long when price breaks above 1d Donchian(20) high with volume > 1.5x 20-period volume average.
Short when price breaks below 1d Donchian(20) low with volume > 1.5x 20-period volume average.
Trailing stop: exit position when price retraces 2.0x ATR(14) from extreme.
Designed to capture strong trending moves while avoiding whipsaws in ranging markets.
Works in both bull and bear markets by trading breakouts in direction of 1d trend.
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
    
    # Get 1d data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Donchian(20) channels
    def donchian_channel(high_vals, low_vals, window):
        upper = pd.Series(high_vals).rolling(window=window, min_periods=window).max().values
        lower = pd.Series(low_vals).rolling(window=window, min_periods=window).min().values
        return upper, lower
    
    donchian_upper_1d, donchian_lower_1d = donchian_channel(high_1d, low_1d, 20)
    
    # Align 1d Donchian channels to 12h timeframe
    donchian_upper_1d_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper_1d)
    donchian_lower_1d_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower_1d)
    
    # Calculate 12h volume 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 12h ATR(14) for trailing stop
    def atr(high_vals, low_vals, close_vals, window):
        tr1 = high_vals - low_vals
        tr2 = np.abs(high_vals - np.roll(close_vals, 1))
        tr3 = np.abs(low_vals - np.roll(close_vals, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # first period
        atr_vals = pd.Series(tr).rolling(window=window, min_periods=window).mean().values
        return atr_vals
    
    atr_14 = atr(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    long_extreme = 0.0
    short_extreme = 0.0
    
    start_idx = 20  # need enough for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_upper_1d_aligned[i]) or 
            np.isnan(donchian_lower_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i]) or 
            np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above 1d Donchian(20) high with volume
            if (close[i] > donchian_upper_1d_aligned[i] and 
                volume_confirmed):
                signals[i] = 0.25
                position = 1
                long_extreme = close[i]
            # Short: price breaks below 1d Donchian(20) low with volume
            elif (close[i] < donchian_lower_1d_aligned[i] and 
                  volume_confirmed):
                signals[i] = -0.25
                position = -1
                short_extreme = close[i]
        
        elif position == 1:
            # Update long extreme
            if close[i] > long_extreme:
                long_extreme = close[i]
            
            # Trailing stop: exit if price retraces 2.0x ATR from extreme
            if close[i] < long_extreme - 2.0 * atr_14[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Update short extreme
            if close[i] < short_extreme:
                short_extreme = close[i]
            
            # Trailing stop: exit if price retraces 2.0x ATR from extreme
            if close[i] > short_extreme + 2.0 * atr_14[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1dDonchian20_Breakout_Volume_ATRTrail"
timeframe = "12h"
leverage = 1.0