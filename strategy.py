#!/usr/bin/env python3
"""
Hypothesis: 1d timeframe with 1w Supertrend filter + 1d Donchian(20) breakout + volume confirmation.
Long when price breaks above 20-day high with 1w Supertrend bullish and volume > 1.5x 20-day volume average.
Short when price breaks below 20-day low with 1w Supertrend bearish and volume > 1.5x 20-day volume average.
Supertrend on weekly timeframe identifies primary trend direction, increasing probability of continuation breakouts.
Designed to work in bull markets (breakout with bullish weekly trend) and bear markets (breakdown with bearish weekly trend).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for Supertrend
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Get 1d data for Donchian channels and volume
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1w Supertrend (10, 3.0)
    def atr(high_vals, low_vals, close_vals, window):
        tr1 = pd.Series(high_vals - low_vals)
        tr2 = pd.Series(np.abs(high_vals - np.roll(close_vals, 1)))
        tr3 = pd.Series(np.abs(low_vals - np.roll(close_vals, 1)))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        tr.iloc[0] = high_vals[0] - low_vals[0]  # first TR
        atr_vals = tr.rolling(window=window, min_periods=window).mean().values
        return atr_vals
    
    def supertrend(high_vals, low_vals, close_vals, atr_period, multiplier):
        atr_vals = atr(high_vals, low_vals, close_vals, atr_period)
        hl2 = (high_vals + low_vals) / 2
        upperband = hl2 + (multiplier * atr_vals)
        lowerband = hl2 - (multiplier * atr_vals)
        
        supertrend = np.zeros_like(close_vals)
        direction = np.ones_like(close_vals)  # 1 for uptrend, -1 for downtrend
        
        supertrend[0] = hl2[0]
        direction[0] = 1
        
        for i in range(1, len(close_vals)):
            if close_vals[i] > upperband[i-1]:
                direction[i] = 1
            elif close_vals[i] < lowerband[i-1]:
                direction[i] = -1
            else:
                direction[i] = direction[i-1]
                if direction[i] == 1 and lowerband[i] < lowerband[i-1]:
                    lowerband[i] = lowerband[i-1]
                if direction[i] == -1 and upperband[i] > upperband[i-1]:
                    upperband[i] = upperband[i-1]
            
            if direction[i] == 1:
                supertrend[i] = lowerband[i]
            else:
                supertrend[i] = upperband[i]
        
        return supertrend, direction
    
    st_1w, st_dir_1w = supertrend(high_1w, low_1w, close_1w, 10, 3.0)
    
    # Calculate 1d Donchian(20) channels
    def donchian_channel(high_vals, low_vals, window):
        upper = pd.Series(high_vals).rolling(window=window, min_periods=window).max().values
        lower = pd.Series(low_vals).rolling(window=window, min_periods=window).min().values
        return upper, lower
    
    donchian_upper, donchian_lower = donchian_channel(high_1d, low_1d, 20)
    
    # Calculate 1d volume 20-period average
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all to primary timeframe (1d)
    st_dir_1w_aligned = align_htf_to_ltf(prices, df_1w, st_dir_1w)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 200  # need enough for Supertrend and Donchian
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(st_dir_1w_aligned[i]) or 
            np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-day average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above 20-day high with weekly bullish trend and volume
            if (close[i] > donchian_upper_aligned[i] and 
                st_dir_1w_aligned[i] == 1 and 
                volume_confirmed):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-day low with weekly bearish trend and volume
            elif (close[i] < donchian_lower_aligned[i] and 
                  st_dir_1w_aligned[i] == -1 and 
                  volume_confirmed):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below 20-day low (opposite side of channel)
            if close[i] < donchian_lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above 20-day high (opposite side of channel)
            if close[i] > donchian_upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1wSupertrend10_3_Donchian20_Breakout_Volume_Confirm"
timeframe = "1d"
leverage = 1.0