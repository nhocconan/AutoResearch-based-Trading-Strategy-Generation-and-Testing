#!/usr/bin/env python3
# 12h_Donchian20_Breakout_1wTrend_Volume
# Hypothesis: Donchian breakout on 12h with 1-week EMA trend filter and volume confirmation.
# Long when 1w trend up and price breaks above upper Donchian(20) with volume > 1.5x average.
# Short when 1w trend down and price breaks below lower Donchian(20) with volume > 1.5x average.
# Uses weekly trend for strong directional bias and Donchian breakouts for entry timing,
# reducing whipsaw in both bull and bear markets. Weekly trend ensures alignment with
# major market cycles, while volume confirmation avoids low-conviction breakouts.

name = "12h_Donchian20_Breakout_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA34 for trend filter
    ema34_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 34:
        ema34_1w[33] = np.mean(close_1w[0:34])
        for i in range(34, len(close_1w)):
            ema34_1w[i] = (close_1w[i] * 2 + ema34_1w[i-1] * 32) / 34
    
    # Align 1w EMA to 12h timeframe
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate Donchian channels (20-period) on 12h data
    upper = np.full_like(high, np.nan)
    lower = np.full_like(low, np.nan)
    
    for i in range(20, len(high)):
        upper[i] = np.max(high[i-20:i])
        lower[i] = np.min(low[i-20:i])
    
    # Volume filter: current volume vs 20-period average
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid_vol = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid_vol] = volume[valid_vol] / vol_ma[valid_vol]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Need 1w EMA and Donchian channels
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(upper[i]) or 
            np.isnan(lower[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 1w trend
        trend_up = close[i] > ema34_1w_aligned[i]
        
        if position == 0:
            # Enter long: 1w trend up + price breaks above upper Donchian + volume confirmation
            if trend_up and close[i] > upper[i] and volume_ratio[i] > 1.5:
                signals[i] = 0.25
                position = 1
            # Enter short: 1w trend down + price breaks below lower Donchian + volume confirmation
            elif not trend_up and close[i] < lower[i] and volume_ratio[i] > 1.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: 1w trend turns down or price breaks below lower Donchian
            if not trend_up or close[i] < lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: 1w trend turns up or price breaks above upper Donchian
            if trend_up or close[i] > upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals