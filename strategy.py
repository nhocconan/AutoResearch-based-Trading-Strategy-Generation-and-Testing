#!/usr/bin/env python3
# 12h_Donchian20_Breakout_1dTrend_With_Volume
# Hypothesis: Donchian channel breakout on 12h timeframe with 1d trend filter (price above/below 200 EMA) and volume confirmation.
# Works in bull markets via breakout above upper band and in bear via breakdown below lower band.
# Volume filter ensures breakouts have conviction. 1d EMA200 filter avoids counter-trend trades.
# Target: 20-40 trades/year on 12h timeframe.

name = "12h_Donchian20_Breakout_1dTrend_With_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def donchian_channels(high, low, window):
    """Calculate Donchian channels: upper = max(high, window), lower = min(low, window)"""
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper = high_series.rolling(window=window, min_periods=window).max()
    lower = low_series.rolling(window=window, min_periods=window).min()
    return upper.values, lower.values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data for trend filter (EMA 200)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Calculate 200 EMA on daily timeframe
    ema_200_1d = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Get 12h data for Donchian channels and volume
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period) on 12h
    upper_20, lower_20 = donchian_channels(high, low, 20)
    
    # Volume filter: current volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Donchian (20) + EMA200 (200) + vol EMA (20)
    start_idx = max(20, 200, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(upper_20[i]) or
            np.isnan(lower_20[i]) or
            np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above upper Donchian AND above 1d EMA200 AND volume confirmation
            if close[i] > upper_20[i] and close[i] > ema_200_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian AND below 1d EMA200 AND volume confirmation
            elif close[i] < lower_20[i] and close[i] < ema_200_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price falls below lower Donchian
            if close[i] < lower_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises above upper Donchian
            if close[i] > upper_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals