#!/usr/bin/env python3
# 6h_1d_obv_momentum_v1
# Hypothesis: On-Balance Volume (OBV) momentum divergence on daily chart with price breakout on 6h.
# Long when price breaks above 6h Donchian high and daily OBV makes higher high (bullish divergence).
# Short when price breaks below 6h Donchian low and daily OBV makes lower low (bearish divergence).
# Exit when price returns to opposite Donchian band or OBV momentum reverses.
# Works in bull markets via breakout confirmation and in bear via divergence at extremes.
# Target: 50-150 total trades over 4 years (12-37/year).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_obv_momentum_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate On-Balance Volume (OBV) on daily
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    obv = np.zeros(len(close_1d))
    obv[0] = volume_1d[0]
    for i in range(1, len(close_1d)):
        if close_1d[i] > close_1d[i-1]:
            obv[i] = obv[i-1] + volume_1d[i]
        elif close_1d[i] < close_1d[i-1]:
            obv[i] = obv[i-1] - volume_1d[i]
        else:
            obv[i] = obv[i-1]
    
    # Align OBV to 6h timeframe
    obv_aligned = align_htf_to_ltf(prices, df_1d, obv)
    
    # Calculate 6-period OBV slope (momentum) on 1d
    obv_slope = np.full(len(obv), np.nan)
    for i in range(6, len(obv)):
        obv_slope[i] = obv[i] - obv[i-6]
    obv_slope_aligned = align_htf_to_ltf(prices, df_1d, obv_slope)
    
    # Calculate 6h Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    high_max = np.full(n, np.nan)
    low_min = np.full(n, np.nan)
    for i in range(n):
        if i >= 19:
            high_max[i] = np.max(high[i-19:i+1])
            low_min[i] = np.min(low[i-19:i+1])
            donchian_high[i] = high_max[i]
            donchian_low[i] = low_min[i]
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(obv_aligned[i]) or 
            np.isnan(obv_slope_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price returns to or below Donchian low OR OBV momentum turns negative
            if close[i] <= donchian_low[i] or obv_slope_aligned[i] < 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to or above Donchian high OR OBV momentum turns positive
            if close[i] >= donchian_high[i] or obv_slope_aligned[i] > 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above Donchian high with bullish OBV momentum
            if (close[i] > donchian_high[i] and 
                obv_slope_aligned[i] > 0):
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below Donchian low with bearish OBV momentum
            elif (close[i] < donchian_low[i] and 
                  obv_slope_aligned[i] < 0):
                position = -1
                signals[i] = -0.25
    
    return signals