#!/usr/bin/env python3
"""
12h_donchian_breakout_1w_trend_volume_v1
Hypothesis: 12h Donchian(20) breakout with weekly EMA25 trend filter and volume confirmation. 
Breakouts above upper band with volume and uptrend go long; breakdowns below lower band with volume and downtrend go short.
Uses weekly trend filter to avoid counter-trend trades, works in both bull and bear markets by following the higher timeframe trend.
Target: 12-37 trades/year on 12h with strict entry conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_breakout_1w_trend_volume_v1"
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
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 25:
        return np.zeros(n)
    
    # Weekly EMA25 for trend filter
    ema25_1w = pd.Series(df_1w['close'].values).ewm(span=25, adjust=False).mean().values
    ema25_1w_aligned = align_htf_to_ltf(prices, df_1w, ema25_1w)
    
    # Daily data for Donchian channels (more stable than 12h alone)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Donchian channels (20-period high/low) from daily data
    high_20 = pd.Series(df_1d['high'].values).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(df_1d['low'].values).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    
    # Volume confirmation: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if data not available
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or
            np.isnan(ema25_1w_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average
        vol_spike = volume[i] > (vol_ma[i] * 1.5)
        
        # Trend filter
        above_ema25 = close[i] > ema25_1w_aligned[i]
        below_ema25 = close[i] < ema25_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price returns to midline or trend turns bearish with volume
            midline = (high_20_aligned[i] + low_20_aligned[i]) / 2
            if close[i] <= midline or (below_ema25 and vol_spike):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to midline or trend turns bullish with volume
            midline = (high_20_aligned[i] + low_20_aligned[i]) / 2
            if close[i] >= midline or (above_ema25 and vol_spike):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Breakout above upper band with volume and uptrend
            if close[i] > high_20_aligned[i] and vol_spike and above_ema25:
                position = 1
                signals[i] = 0.25
            # Breakdown below lower band with volume and downtrend
            elif close[i] < low_20_aligned[i] and vol_spike and below_ema25:
                position = -1
                signals[i] = -0.25
    
    return signals