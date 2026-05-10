#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_12hTrend_Volume
Hypothesis: 6h Donchian(20) breakout in direction of 12h EMA20 trend, with volume confirmation.
Donchian channels capture breakout momentum, 12h EMA20 filters for intermediate trend direction,
volume confirms breakout strength. Works in bull/bear by following higher timeframe trend.
Target: 20-40 trades/year on 6h to avoid fee drag.
"""

name = "6h_Donchian20_Breakout_12hTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 1:
        return np.zeros(n)
    
    # 12h EMA20 for trend filter
    ema_20 = pd.Series(df_12h['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_aligned = align_htf_to_ltf(prices, df_12h, ema_20)
    
    # Calculate Donchian channels (20-period high/low)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian upper/lower (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Donchian (20) and EMA20
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_20_aligned[i]) or 
            np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: above 12h EMA20 (uptrend) AND price breaks above Donchian high with volume
            if close[i] > ema_20_aligned[i] and high[i] > donchian_high[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: below 12h EMA20 (downtrend) AND price breaks below Donchian low with volume
            elif close[i] < ema_20_aligned[i] and low[i] < donchian_low[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below Donchian low OR trend turns bearish
            if low[i] < donchian_low[i] or close[i] < ema_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Donchian high OR trend turns bullish
            if high[i] > donchian_high[i] or close[i] > ema_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals