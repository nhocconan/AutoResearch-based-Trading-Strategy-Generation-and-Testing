#!/usr/bin/env python3
"""
Hypothesis: 4h timeframe with 1d Donchian(20) breakout + volume confirmation + trend filter (price > 4h EMA34).
Long when price breaks above 1d Donchian upper channel with volume confirmation and price > 4h EMA34 (uptrend).
Short when price breaks below 1d Donchian lower channel with volume confirmation and price < 4h EMA34 (downtrend).
Exit when price returns to the 1d Donchian midpoint (mean reversion to channel center).
Designed to capture medium-term breakouts with institutional volume while avoiding false breakouts in choppy markets.
Uses 1d timeframe for structure (reduces noise) and 4h for entry timing and trend filter.
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
    
    # Get 1d data for Donchian channel calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d Donchian(20) channels
    lookback = 20
    upper_1d = pd.Series(high_1d).rolling(window=lookback, min_periods=lookback).max().values
    lower_1d = pd.Series(low_1d).rolling(window=lookback, min_periods=lookback).min().values
    mid_1d = (upper_1d + lower_1d) / 2.0
    
    # Calculate 4h EMA34 for trend filter
    close_s = pd.Series(close)
    ema34 = close_s.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 4h volume 20-period average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d Donchian levels to 4h timeframe
    upper_1d_aligned = align_htf_to_ltf(prices, df_1d, upper_1d)
    lower_1d_aligned = align_htf_to_ltf(prices, df_1d, lower_1d)
    mid_1d_aligned = align_htf_to_ltf(prices, df_1d, mid_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # need enough for EMA34 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_1d_aligned[i]) or 
            np.isnan(lower_1d_aligned[i]) or 
            np.isnan(mid_1d_aligned[i]) or 
            np.isnan(ema34[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above 1d Donchian upper with volume and uptrend (price > EMA34)
            if (close[i] > upper_1d_aligned[i] and 
                volume_confirmed and 
                close[i] > ema34[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d Donchian lower with volume and downtrend (price < EMA34)
            elif (close[i] < lower_1d_aligned[i] and 
                  volume_confirmed and 
                  close[i] < ema34[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to or below 1d Donchian midpoint
            if close[i] <= mid_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to or above 1d Donchian midpoint
            if close[i] >= mid_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1dDonchian20_Breakout_Volume_EMA34_Trend"
timeframe = "4h"
leverage = 1.0