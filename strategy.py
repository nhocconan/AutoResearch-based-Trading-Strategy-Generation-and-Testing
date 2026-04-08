#!/usr/bin/env python3
# 4h_Donchian_breakout_1d_trend_volume_v1
# Hypothesis: Breakout from 1-day Donchian channels with volume confirmation and 1-week trend filter.
# Works in bull markets by buying breakouts above upper channel; in bear markets by shorting breakdowns below lower channel.
# Uses volume surge to confirm institutional participation and reduce false signals.
# Weekly trend filter ensures we only trade in the direction of the higher timeframe trend.
# Target: 75-200 total trades over 4 years (19-50/year).

name = "4h_Donchian_breakout_1d_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1D Donchian channels (20-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 20-period Donchian channels
    upper_channel = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_channel = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian channels to 4h
    upper_channel_aligned = align_htf_to_ltf(prices, df_1d, upper_channel)
    lower_channel_aligned = align_htf_to_ltf(prices, df_1d, lower_channel)
    
    # 1W trend filter: EMA crossover
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema25_1w = pd.Series(close_1w).ewm(span=25, adjust=False, min_periods=25).mean().values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Weekly trend: 1 if EMA25 > EMA50 (bullish), -1 if EMA25 < EMA50 (bearish)
    trend_1w = np.where(ema25_1w > ema50_1w, 1, -1)
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_surge = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    
    # Start after sufficient warmup
    start_idx = max(40, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_channel_aligned[i]) or np.isnan(lower_channel_aligned[i]) or 
            np.isnan(trend_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions: price breaks above upper Donchian channel + volume surge + weekly bullish trend
        if (close[i] > upper_channel_aligned[i] and 
            vol_surge[i] and 
            trend_1w_aligned[i] == 1):
            signals[i] = 0.25
        
        # Short conditions: price breaks below lower Donchian channel + volume surge + weekly bearish trend
        elif (close[i] < lower_channel_aligned[i] and 
              vol_surge[i] and 
              trend_1w_aligned[i] == -1):
            signals[i] = -0.25
        
        # Otherwise flat
        else:
            signals[i] = 0.0
    
    return signals