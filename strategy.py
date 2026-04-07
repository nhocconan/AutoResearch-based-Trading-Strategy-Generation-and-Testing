#!/usr/bin/env python3
"""
4h_donchian_breakout_12h_trend_volume_v1
Hypothesis: Donchian(20) breakout on 4h with 12h EMA20 trend filter and volume confirmation.
In trending markets, breakouts of the 20-bar high/low with volume and trend alignment
capture momentum. In ranging markets, the trend filter prevents false breakouts.
Volume confirmation ensures breakouts are supported by participation.
Works in both bull and bear by adapting to trend via 12h EMA filter.
Target: 20-50 trades/year on 4h with strict entry conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_12h_trend_volume_v1"
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
    
    # 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # 12h EMA20 for trend filter
    ema20_12h = pd.Series(df_12h['close'].values).ewm(span=20, adjust=False).mean().values
    ema20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema20_12h)
    
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
            np.isnan(ema20_12h_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average
        vol_spike = volume[i] > (vol_ma[i] * 1.5)
        
        # Trend filter
        above_ema20 = close[i] > ema20_12h_aligned[i]
        below_ema20 = close[i] < ema20_12h_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian low or trend turns bearish with volume
            if close[i] < low_min[i] or (below_ema20 and vol_spike):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high or trend turns bullish with volume
            if close[i] > high_max[i] or (above_ema20 and vol_spike):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat, look for entry
            # Breakout entry with volume and trend confirmation
            if close[i] > high_max[i] and vol_spike and above_ema20:
                # Bullish breakout with volume and uptrend
                position = 1
                signals[i] = 0.30
            elif close[i] < low_min[i] and vol_spike and below_ema20:
                # Bearish breakout with volume and downtrend
                position = -1
                signals[i] = -0.30
    
    return signals