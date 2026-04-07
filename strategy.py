#!/usr/bin/env python3
"""
4h_donchian_breakout_1d_trend_volume_v3
Hypothesis: Donchian(20) breakouts with 1d trend filter and volume confirmation work in both bull and bear markets.
Breakouts capture momentum; 1d trend filter avoids counter-trend trades; volume confirms institutional participation.
Designed for 4h timeframe with tight entry conditions to target 20-40 trades/year, minimizing fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_1d_trend_volume_v3"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 1d trend filter: EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_50d = pd.Series(close_1d).ewm(span=50, min_periods=50).mean().values
    ema_50d_aligned = align_htf_to_ltf(prices, df_1d, ema_50d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] == 0 or 
            np.isnan(ema_50d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below 20-period low or trend turns bearish
            if close[i] < lowest_low[i] or ema_50d_aligned[i] < ema_50d_aligned[i-1]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above 20-period high or trend turns bullish
            if close[i] > highest_high[i] or ema_50d_aligned[i] > ema_50d_aligned[i-1]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price breaks above 20-period high, volume confirmed, bullish 1d trend
            if close[i] > highest_high[i] and volume[i] > vol_ma[i] and ema_50d_aligned[i] > ema_50d_aligned[i-1]:
                position = 1
                signals[i] = 0.25
            # Short: price breaks below 20-period low, volume confirmed, bearish 1d trend
            elif close[i] < lowest_low[i] and volume[i] > vol_ma[i] and ema_50d_aligned[i] < ema_50d_aligned[i-1]:
                position = -1
                signals[i] = -0.25
    
    return signals