#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_VolumeConfirm_TrendFilter
Hypothesis: On 4h timeframe, enter long when price breaks above Donchian(20) upper band with volume > 1.5x 20-period average, confirmed by 1d EMA50 uptrend; enter short when price breaks below Donchian(20) lower band with volume > 1.5x 20-period average, confirmed by 1d EMA50 downtrend. Exit on opposite breakout or when trend fails. Uses volume confirmation and trend filter to reduce false breakouts, targeting 20-40 trades/year to minimize fee drag. Works in bull/bear via 1d trend filter.
"""

name = "4h_Donchian20_Breakout_VolumeConfirm_TrendFilter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 4h data for Donchian channels and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian(20) channels
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Donchian(20) and volume MA(20)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: 1d EMA50 direction
        uptrend_1d = close[i] > ema50_1d_aligned[i]
        downtrend_1d = close[i] < ema50_1d_aligned[i]
        
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = volume[i] > vol_ma20[i] * 1.5
        
        if position == 0:
            # Long: break above Donchian upper with volume and uptrend
            if close[i] > highest_high[i] and volume_filter and uptrend_1d:
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian lower with volume and downtrend
            elif close[i] < lowest_low[i] and volume_filter and downtrend_1d:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: break below Donchian lower or trend fails
            if close[i] < lowest_low[i] or not uptrend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: break above Donchian upper or trend fails
            if close[i] > highest_high[i] or not downtrend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals