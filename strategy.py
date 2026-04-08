#!/usr/bin/env python3
# 4h_1d_donchian_breakout_volume_confirm_v1
# Hypothesis: 4h Donchian(20) breakout with 1d trend filter (price > SMA50) and volume confirmation.
# Long when 4h close breaks above 20-bar high, price > 1d SMA50, and volume > 1.5x 20-bar average.
# Short when 4h close breaks below 20-bar low, price < 1d SMA50, and volume > 1.5x 20-bar average.
# Designed for 20-50 trades/year on 4h timeframe to balance signal quality and frequency.
# Works in bull markets via upward breakouts and bear markets via downward breakdowns.
# Uses 1d SMA for trend filter to avoid counter-trend trades.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_donchian_breakout_volume_confirm_v1"
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
    
    # 4h Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 4h volume confirmation (20-bar average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 1d SMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    sma_50_1d = pd.Series(df_1d['close']).rolling(window=50, min_periods=50).mean().values
    sma_50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 50  # Ensure SMA50 and Donchian are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(vol_ma_20[i]) or np.isnan(sma_50_1d_aligned[i]):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 1.5 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: close below 20-bar low
            if close[i] < low_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: close above 20-bar high
            if close[i] > high_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: break above 20-bar high, price > 1d SMA50, volume surge
            if close[i] > high_20[i] and close[i] > sma_50_1d_aligned[i] and vol_surge:
                position = 1
                signals[i] = 0.25
            # Short entry: break below 20-bar low, price < 1d SMA50, volume surge
            elif close[i] < low_20[i] and close[i] < sma_50_1d_aligned[i] and vol_surge:
                position = -1
                signals[i] = -0.25
    
    return signals