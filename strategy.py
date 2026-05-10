#!/usr/bin/env python3
# 4h_1d_Donchian_Breakout_VolumeTrend
# Hypothesis: 4h Donchian channel breakouts filtered by 1d EMA50 trend and volume surge.
# Donchian captures price extremes; EMA50 determines trend direction; volume surge confirms strength.
# Designed for low trade frequency (<30/year) to minimize fee drift and work in bull/bear markets.

name = "4h_1d_Donchian_Breakout_VolumeTrend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 4h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Donchian (20) + volume MA (20) + EMA (50)
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend from 1d EMA50
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
        uptrend = close_1d_aligned[i] > ema_50_1d_aligned[i]
        downtrend = close_1d_aligned[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation
        volume_surge = volume[i] > 1.5 * vol_ma[i]
        
        # Donchian breakout signals
        breakout_up = close[i] > highest_high[i-1]
        breakout_down = close[i] < lowest_low[i-1]
        
        if position == 0:
            # Long: Donchian breakout up with volume surge and 1d uptrend
            if breakout_up and volume_surge and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakdown down with volume surge and 1d downtrend
            elif breakout_down and volume_surge and downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Donchian breakdown OR trend changes
            if close[i] < lowest_low[i-1] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Donchian breakout up OR trend changes
            if close[i] > highest_high[i-1] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals