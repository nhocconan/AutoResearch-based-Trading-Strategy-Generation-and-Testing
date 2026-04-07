#!/usr/bin/env python3
"""
4h_donchian_breakout_1d_trend_volume_v2
Hypothesis: Donchian(20) breakouts on 4h with volume confirmation and 1d EMA50 trend filter.
In trending markets, breakouts with volume continue the trend. In ranging markets, false breakouts fail without volume.
Works in both bull and bear by adapting to trend filter - only take breakouts in direction of higher timeframe trend.
Target: 20-50 trades/year on 4h with strict entry conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_1d_trend_volume_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Daily EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Donchian channels (20-period) on 4h
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if data not available
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average
        vol_spike = volume[i] > (vol_ma[i] * 1.5)
        
        # Trend filter
        above_ema50 = close[i] > ema50_1d_aligned[i]
        below_ema50 = close[i] < ema50_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price returns to Donchian middle or trend turns bearish with volume
            donchian_mid = (high_max[i] + low_min[i]) / 2
            if close[i] <= donchian_mid or (below_ema50 and vol_spike):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to Donchian middle or trend turns bullish with volume
            donchian_mid = (high_max[i] + low_min[i]) / 2
            if close[i] >= donchian_mid or (above_ema50 and vol_spike):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Breakout with volume and trend alignment
            if high[i] > high_max[i] and vol_spike and above_ema50:
                # Bullish breakout with volume and uptrend
                position = 1
                signals[i] = 0.25
            elif low[i] < low_min[i] and vol_spike and below_ema50:
                # Bearish breakout with volume and downtrend
                position = -1
                signals[i] = -0.25
    
    return signals