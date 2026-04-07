#!/usr/bin/env python3
"""
12h_donchian_breakout_1d_trend_volume_v1
Hypothesis: On 12h timeframe, use Donchian channel breakout for entry/exit, filtered by 1d trend (EMA200) and volume confirmation. This strategy captures medium-term trends with minimal trades (target: 50-150 total over 4 years). Donchian breakouts provide clear entry/exit signals, EMA200 filter ensures trend alignment, and volume confirmation reduces false breakouts. Works in both bull and bear markets by following the dominant trend as defined by EMA200.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_breakout_1d_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channel parameters
    donchian_period = 20
    
    # Calculate Donchian channels
    highest_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lowest_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # EMA200 for 1d trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):
        # Skip if data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema200_1d_aligned[i]) or np.isnan(vol_ma[i]) or
            np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower band
            if close[i] < lowest_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper band
            if close[i] > highest_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average
            volume_ok = volume[i] > 1.5 * vol_ma[i]
            
            if volume_ok:
                # Long entry: price breaks above Donchian upper band AND above EMA200 (uptrend)
                if close[i] > highest_high[i] and close[i] > ema200_1d_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price breaks below Donchian lower band AND below EMA200 (downtrend)
                elif close[i] < lowest_low[i] and close[i] < ema200_1d_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals