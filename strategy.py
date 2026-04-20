#!/usr/bin/env python3
# 4h_Donchian20_VolumeSurge_TrendFilter
# Hypothesis: 4h Donchian breakout with volume surge and trend filter captures strong directional moves.
# Long: Close breaks above Donchian upper (20) + volume > 2x avg + price > EMA50 (trend)
# Short: Close breaks below Donchian lower (20) + volume > 2x avg + price < EMA50 (trend)
# Exit: Opposite Donchian break or volume drop below average
# Volume surge filters weak breakouts. Trend filter avoids counter-trend whipsaws.
# Target: 75-200 total trades over 4 years (19-50/year) to avoid fee drag.

name = "4h_Donchian20_VolumeSurge_TrendFilter"
timeframe = "4h"
leverage = 1.0

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
    
    # Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # EMA50 for trend filter
    close_s = pd.Series(close)
    ema50 = close_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume average (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(ema50[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above Donchian upper + volume surge + uptrend
            if (close[i] > high_roll[i] and 
                volume[i] > (volume_ma[i] * 2.0) and 
                close[i] > ema50[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian lower + volume surge + downtrend
            elif (close[i] < low_roll[i] and 
                  volume[i] > (volume_ma[i] * 2.0) and 
                  close[i] < ema50[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below Donchian lower or volume drops
            if close[i] < low_roll[i] or volume[i] < volume_ma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above Donchian upper or volume drops
            if close[i] > high_roll[i] or volume[i] < volume_ma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals