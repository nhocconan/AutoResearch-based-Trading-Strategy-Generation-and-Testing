#!/usr/bin/env python3
# 4h_1d_1w_volume_price_action_v1
# Hypothesis: Trade breakouts of daily price channels (Donchian) with volume confirmation and weekly trend filter.
# Uses daily Donchian channels for medium-term breakouts, volume surge to confirm strength, and weekly trend filter (price > weekly EMA50) to align with higher timeframe momentum.
# Long when price breaks above upper Donchian channel with volume surge and weekly uptrend (price > weekly EMA50).
# Short when price breaks below lower Donchian channel with volume surge and weekly downtrend (price < weekly EMA50).
# Designed for 4h timeframe to target 20-50 trades/year (80-200 total over 4 years).
# Weekly trend filter ensures alignment with higher timeframe momentum, working in both bull and bear markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_1w_volume_price_action_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Donchian channels (20-day)
    upper_donchian = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_donchian = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Weekly trend filter: EMA50
    close_1w = get_htf_data(prices, '1w')['close'].values
    ema50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align daily and weekly data to 4h timeframe
    upper_donchian_aligned = align_htf_to_ltf(prices, df_1d, upper_donchian)
    lower_donchian_aligned = align_htf_to_ltf(prices, df_1d, lower_donchian)
    ema50_aligned = align_htf_to_ltf(prices, get_htf_data(prices, '1w'), ema50)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 60  # Ensure Donchian and EMA50 are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_donchian_aligned[i]) or np.isnan(lower_donchian_aligned[i]) or 
            np.isnan(ema50_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 1.5 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: price breaks below lower Donchian channel
            if close[i] < lower_donchian_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above upper Donchian channel
            if close[i] > upper_donchian_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above upper Donchian channel with volume surge and weekly uptrend
            if (close[i] > upper_donchian_aligned[i] and vol_surge and 
                close[i] > ema50_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below lower Donchian channel with volume surge and weekly downtrend
            elif (close[i] < lower_donchian_aligned[i] and vol_surge and 
                  close[i] < ema50_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals