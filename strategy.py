#!/usr/bin/env python3
"""
4h_Donchian20_VolumeSurge_TrendFilter
Hypothesis: 4h Donchian(20) breakout with volume surge (>2x average) and EMA50 trend filter.
Works in bull/bear: long only when price > EMA50, short only when price < EMA50.
Volume surge filters for institutional participation, reducing false breakouts.
Target: 20-50 trades/year per symbol (<200 total over 4 years) to minimize fee drag.
"""

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
    
    # EMA50 for trend filter
    close_series = pd.Series(close)
    ema50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper = high_series.rolling(window=20, min_periods=20).max().values
    lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume surge: >2x 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_surge = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure EMA50 and Donchian are valid
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(ema50[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above upper Donchian with volume surge and uptrend (price > EMA50)
            if (close[i] > upper[i] and 
                volume_surge[i] and 
                close[i] > ema50[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian with volume surge and downtrend (price < EMA50)
            elif (close[i] < lower[i] and 
                  volume_surge[i] and 
                  close[i] < ema50[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below lower Donchian or trend fails (price < EMA50)
            if (close[i] < lower[i]) or (close[i] < ema50[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above upper Donchian or trend fails (price > EMA50)
            if (close[i] > upper[i]) or (close[i] > ema50[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals