#!/usr/bin/env python3
"""
12h_Donchian_Breakout_Volume_Trend
Hypothesis: Donchian channel breakouts with volume confirmation and 1d trend filter.
In bull markets, price breaks above upper Donchian band in uptrend; in bear markets,
breaks below lower band in downtrend. Volume confirms institutional participation.
1d EMA50 trend filter avoids counter-trend trades. Target: 20-40 trades/year
(80-160 total over 4 years) to minimize fee drag.
"""

name = "12h_Donchian_Breakout_Volume_Trend"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channel (20-period)
    donchian_window = 20
    upper_donchian = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    lower_donchian = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # Volume confirmation (24-period MA on 12h = ~12 days)
    volume_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Donchian (20), volume MA (24), 1d EMA50 (50)
    start_idx = max(donchian_window, 24, 50)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(upper_donchian[i]) or 
            np.isnan(lower_donchian[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Higher timeframe trend filter
        uptrend_1d = close[i] > ema_50_1d_aligned[i]
        downtrend_1d = close[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation (>1.8x average volume)
        volume_confirm = volume[i] > volume_ma[i] * 1.8
        
        if position == 0:
            # Long entry: uptrend + price breaks above upper Donchian + volume
            if uptrend_1d and close[i] > upper_donchian[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: downtrend + price breaks below lower Donchian + volume
            elif downtrend_1d and close[i] < lower_donchian[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend breaks or price re-enters below upper Donchian
            if not uptrend_1d or close[i] < lower_donchian[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend breaks or price re-enters above lower Donchian
            if not downtrend_1d or close[i] > upper_donchian[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals