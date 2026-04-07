#!/usr/bin/env python3
"""
12h_donchian_breakout_1d_trend_volume_v1
Hypothesis: 12h Donchian(20) breakout with 1d EMA20 trend filter and volume confirmation.
Breakouts above/below 20-period high/low with volume and trend alignment capture momentum.
Works in bull markets via breakout continuation and in bear via mean-reversion at channel edges.
Target: 15-35 trades/year on 12h with strict entry conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_breakout_1d_trend_volume_v1"
timeframe = "12h"
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
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily EMA20 for trend filter
    ema20_1d = pd.Series(df_1d['close'].values).ewm(span=20, adjust=False).mean().values
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # 12h Donchian channels (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(lookback, n):
        # Skip if data not available
        if (np.isnan(ema20_1d_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average
        vol_spike = volume[i] > (vol_ma[i] * 1.5)
        
        # Trend filter
        above_ema20 = close[i] > ema20_1d_aligned[i]
        below_ema20 = close[i] < ema20_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price returns to Donchian midpoint or trend turns bearish with volume
            midpoint = (highest_high[i] + lowest_low[i]) / 2
            if close[i] <= midpoint or (below_ema20 and vol_spike):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to Donchian midpoint or trend turns bullish with volume
            midpoint = (highest_high[i] + lowest_low[i]) / 2
            if close[i] >= midpoint or (above_ema20 and vol_spike):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Breakout with volume and trend alignment
            if close[i] > highest_high[i] and vol_spike and above_ema20:
                # Bullish breakout with volume and trend
                position = 1
                signals[i] = 0.25
            elif close[i] < lowest_low[i] and vol_spike and below_ema20:
                # Bearish breakout with volume and trend
                position = -1
                signals[i] = -0.25
    
    return signals