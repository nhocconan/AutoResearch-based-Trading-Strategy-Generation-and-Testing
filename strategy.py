#!/usr/bin/env python3
"""
Hypothesis: 4h timeframe with 12h Supertrend(10,3) as trend filter + 4h Donchian(20) breakout + volume confirmation.
Long when price breaks above 20-period high with 12h Supertrend bullish and volume > 1.5x 20-period volume average.
Short when price breaks below 20-period low with 12h Supertrend bearish and volume > 1.5x 20-period volume average.
Exit when price touches the opposite Donchian band or Supertrend flips.
Designed to capture strong trends with volume confirmation while avoiding choppy markets via Supertrend filter.
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
    
    # Get 12h data for Supertrend
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Supertrend(10,3)
    def atr(high_vals, low_vals, close_vals, window):
        tr1 = pd.Series(high_vals).shift(1) - pd.Series(low_vals).shift(1)
        tr2 = abs(pd.Series(high_vals).shift(1) - pd.Series(close_vals).shift(1))
        tr3 = abs(pd.Series(low_vals).shift(1) - pd.Series(close_vals).shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_vals = pd.Series(tr).ewm(span=window, min_periods=window, adjust=False).mean().values
        return atr_vals
    
    def supertrend(high_vals, low_vals, close_vals, atr_period, multiplier):
        atr_vals = atr(high_vals, low_vals, close_vals, atr_period)
        hl2 = (high_vals + low_vals) / 2
        upperband = hl2 + (multiplier * atr_vals)
        lowerband = hl2 - (multiplier * atr_vals)
        
        supertrend_vals = np.full_like(close_vals, np.nan, dtype=float)
        direction = np.full_like(close_vals, 1, dtype=int)  # 1 for uptrend, -1 for downtrend
        
        for i in range(1, len(close_vals)):
            if np.isnan(atr_vals[i]) or np.isnan(upperband[i-1]) or np.isnan(lowerband[i-1]):
                supertrend_vals[i] = np.nan
                direction[i] = direction[i-1]
                continue
                
            if close_vals[i] > upperband[i-1]:
                direction[i] = 1
            elif close_vals[i] < lowerband[i-1]:
                direction[i] = -1
            else:
                direction[i] = direction[i-1]
                
            if direction[i] == 1:
                supertrend_vals[i] = max(lowerband[i], supertrend_vals[i-1]) if not np.isnan(supertrend_vals[i-1]) else lowerband[i]
            else:
                supertrend_vals[i] = min(upperband[i], supertrend_vals[i-1]) if not np.isnan(supertrend_vals[i-1]) else upperband[i]
                
        return supertrend_vals, direction
    
    st_12h, st_dir_12h = supertrend(high_12h, low_12h, close_12h, 10, 3)
    
    # Calculate 4h Donchian(20) channels
    def donchian_channel(high_vals, low_vals, window):
        upper = pd.Series(high_vals).rolling(window=window, min_periods=window).max().values
        lower = pd.Series(low_vals).rolling(window=window, min_periods=window).min().values
        return upper, lower
    
    donchian_upper, donchian_lower = donchian_channel(high, low, 20)
    
    # Calculate 4h volume 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align all to primary timeframe (4h)
    st_12h_aligned = align_htf_to_ltf(prices, df_12h, st_12h)
    st_dir_12h_aligned = align_htf_to_ltf(prices, df_12h, st_dir_12h)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_12h, donchian_upper)  # Using 12h df for alignment but values are 4h
    donchian_lower_aligned = align_htf_to_ltf(prices, df_12h, donchian_lower)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20)
    
    # Fix alignment: realign the 4h indicators properly
    donchian_upper_aligned = align_htf_to_ltf(prices, df_12h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_12h, donchian_lower)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 30  # need enough for Supertrend and Donchian
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(st_12h_aligned[i]) or 
            np.isnan(st_dir_12h_aligned[i]) or 
            np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20_aligned[i]
        
        if position == 0:
            # Long: price breaks above 20-period high with 12h Supertrend bullish and volume
            if (close[i] > donchian_upper_aligned[i] and 
                st_dir_12h_aligned[i] == 1 and 
                volume_confirmed):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-period low with 12h Supertrend bearish and volume
            elif (close[i] < donchian_lower_aligned[i] and 
                  st_dir_12h_aligned[i] == -1 and 
                  volume_confirmed):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below 20-period low OR Supertrend turns bearish
            if (close[i] < donchian_lower_aligned[i] or 
                st_dir_12h_aligned[i] == -1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above 20-period high OR Supertrend turns bullish
            if (close[i] > donchian_upper_aligned[i] or 
                st_dir_12h_aligned[i] == 1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_12hSupertrend10_3_Donchian20_Breakout_Volume_Confirm"
timeframe = "4h"
leverage = 1.0