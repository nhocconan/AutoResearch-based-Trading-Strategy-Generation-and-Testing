#!/usr/bin/env python3
# 4h_1d_donchian_breakout_volume_v1
# Hypothesis: Trade 4-hour Donchian channel breakouts with daily volume confirmation and trend filter.
# Long when price breaks above 20-period Donchian high with volume > 1.5x 20-bar average and price > daily EMA50.
# Short when price breaks below 20-period Donchian low with volume > 1.5x 20-bar average and price < daily EMA50.
# Exit when price returns to the Donchian midpoint (average of 20-period high and low).
# Designed for 4h timeframe to target 20-50 trades/year (80-200 total over 4 years).
# Daily trend filter ensures alignment with higher timeframe momentum, working in both bull and bear markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_donchian_breakout_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # EMA50 for daily trend
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Donchian channels (20-period) on 4h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 50  # Ensure EMA50 and Donchian are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 1.5 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: price returns to Donchian midpoint
            if close[i] <= donchian_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to Donchian midpoint
            if close[i] >= donchian_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above Donchian high with volume surge and daily uptrend
            if (close[i] > highest_high[i] and vol_surge and 
                close[i] > ema50_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below Donchian low with volume surge and daily downtrend
            elif (close[i] < lowest_low[i] and vol_surge and 
                  close[i] < ema50_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals