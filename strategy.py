#!/usr/bin/env python3
# 12h_donchian_breakout_1w_trend_volume_v1
# Hypothesis: Uses Donchian breakout on 1w for trend direction, with 12h price action and volume confirmation for entry.
# Enters long when price breaks above 1w Donchian upper band and 12h volume > 1.5x average, short when breaks below lower band with high volume.
# Exits when price returns to the 1w Donchian midpoint. Designed for 15-25 trades/year on 12h to avoid fee drag.
# Works in bull/bear via trend-following with volume confirmation to avoid false breakouts.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_breakout_1w_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-week data for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Donchian channels (20-period)
    donchian_high = np.full_like(high_1w, np.nan)
    donchian_low = np.full_like(low_1w, np.nan)
    donchian_mid = np.full_like(close_1w, np.nan)
    
    for i in range(20, len(high_1w)):
        donchian_high[i] = np.max(high_1w[i-20:i])
        donchian_low[i] = np.min(low_1w[i-20:i])
        donchian_mid[i] = (donchian_high[i] + donchian_low[i]) / 2
    
    # 12h average volume for confirmation
    vol_ma = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            vol_ma[i] = np.mean(volume[max(0, i-19):i+1]) if i > 0 else volume[i]
        else:
            vol_ma[i] = np.mean(volume[i-20:i])
    
    # Align 1w Donchian channels and volume MA to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1w, donchian_mid)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1w, vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        if np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or np.isnan(vol_ma_aligned[i]):
            if position != 0:
                pass
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma_aligned[i] if vol_ma_aligned[i] > 0 else 0
        
        if position == 1:  # Long position
            # Exit: price returns to Donchian midpoint
            if close[i] <= donchian_mid_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to Donchian midpoint
            if close[i] >= donchian_mid_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above Donchian high with volume confirmation
            if close[i] > donchian_high_aligned[i] and vol_ratio > 1.5:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below Donchian low with volume confirmation
            elif close[i] < donchian_low_aligned[i] and vol_ratio > 1.5:
                position = -1
                signals[i] = -0.25
    
    return signals