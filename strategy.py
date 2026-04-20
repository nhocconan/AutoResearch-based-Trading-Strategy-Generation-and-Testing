#!/usr/bin/env python3
"""
1h_Donchian_Breakout_Volume_Trend_Filter
Hypothesis: Trade 1h Donchian channel breakouts with volume confirmation and 4h trend filter.
Long when price breaks above 20-period Donchian high with volume spike and 4h uptrend; short when breaks below 20-period Donchian low with volume spike and 4h downtrend.
Uses 4h EMA50 for trend direction and volume > 1.8x 20-period average for confirmation.
Designed for 1h timeframe to capture medium-term moves while reducing noise via multi-timeframe confirmation.
Target: 60-150 total trades over 4 years (15-37/year) with position size 0.20.
Works in bull/bear: 4h trend filter avoids counter-trend trades, volume filter reduces false breakouts.
"""

name = "1h_Donchian_Breakout_Volume_Trend_Filter"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema50_4h = np.full_like(close_4h, np.nan)
    if len(close_4h) >= 50:
        multiplier = 2.0 / (50 + 1)
        ema50_4h[49] = np.mean(close_4h[:50])
        for i in range(50, len(close_4h)):
            ema50_4h[i] = multiplier * close_4h[i] + (1 - multiplier) * ema50_4h[i-1]
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Calculate 1h Donchian channels (20-period)
    donchian_high = np.full_like(high, np.nan)
    donchian_low = np.full_like(low, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Calculate volume filter (volume > 1.8x 20-period average)
    vol_ma20 = np.full_like(volume, np.nan)
    for i in range(20, n):
        vol_ma20[i] = np.mean(volume[i-20:i])
    volume_filter = volume > (1.8 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema50_4h_aligned[i]) or np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high with volume filter AND 4h uptrend
            if close[i] > donchian_high[i] and volume_filter[i] and close[i] > ema50_4h_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below Donchian low with volume filter AND 4h downtrend
            elif close[i] < donchian_low[i] and volume_filter[i] and close[i] < ema50_4h_aligned[i]:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below Donchian low OR 4h trend turns down
            if close[i] < donchian_low[i] or close[i] < ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: price breaks above Donchian high OR 4h trend turns up
            if close[i] > donchian_high[i] or close[i] > ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals