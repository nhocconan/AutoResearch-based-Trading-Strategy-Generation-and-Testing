#!/usr/bin/env python3
# 4h_12h_1d_donchian_breakout_volume_filter_v1
# Hypothesis: 4-hour Donchian channel breakout (20-period) with 12h volume confirmation and 1d trend filter.
# Long when price breaks above Donchian upper band + 12h volume > 1.5x 20-period average + 1d close above 200 EMA.
# Short when price breaks below Donchian lower band + 12h volume > 1.5x average + 1d close below 200 EMA.
# Uses tight entry conditions to limit trades (~20-30 per year) and avoid fee drag.
# Works in bull/bear via multi-timeframe alignment and volume confirmation.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_1d_donchian_breakout_volume_filter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h Donchian channel (20-period)
    donchian_upper = np.full(n, np.nan)
    donchian_lower = np.full(n, np.nan)
    for i in range(20, n):
        donchian_upper[i] = np.max(high[i-20:i+1])
        donchian_lower[i] = np.min(low[i-20:i+1])
    
    # 12h data for volume confirmation
    df_12h = get_htf_data(prices, '12h')
    volume_12h = df_12h['volume'].values
    # 20-period average volume on 12h
    avg_volume_12h = np.full(len(df_12h), np.nan)
    for i in range(20, len(df_12h)):
        avg_volume_12h[i] = np.mean(volume_12h[i-20:i+1])
    avg_volume_12h_aligned = align_htf_to_ltf(prices, df_12h, avg_volume_12h)
    
    # 1d data for trend filter (200 EMA)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(20, 20)  # Ensure Donchian and averages are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(avg_volume_12h_aligned[i]) or np.isnan(ema200_1d_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x 20-period average
        # Find the corresponding 12h bar index for current 4h bar
        # Since we aligned the 12h average volume, we can use it directly
        vol_confirmed = volume_12h[min(i // 3, len(volume_12h)-1)] > 1.5 * avg_volume_12h[min(i // 3, len(avg_volume_12h)-1)] if i // 3 < len(volume_12h) else False
        
        # Simpler: use the aligned average volume and current 12h volume (need to get current 12h volume)
        # Instead, we'll use the volume ratio from the aligned data
        # Get current 12h volume by accessing the volume_12h array at the correct index
        vol_12h_idx = i // 3  # 3x 4h bars per 12h bar
        if vol_12h_idx < len(volume_12h):
            vol_12h_current = volume_12h[vol_12h_idx]
            vol_12h_avg = avg_volume_12h[vol_12h_idx] if vol_12h_idx < len(avg_volume_12h) else np.nan
            vol_confirmed = not np.isnan(vol_12h_avg) and vol_12h_current > 1.5 * vol_12h_avg
        else:
            vol_confirmed = False
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian lower band
            if close[i] < donchian_lower[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian upper band
            if close[i] > donchian_upper[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above Donchian upper band + volume confirmation + 1d uptrend
            if (close[i] > donchian_upper[i] and 
                vol_confirmed and 
                close[i] > ema200_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below Donchian lower band + volume confirmation + 1d downtrend
            elif (close[i] < donchian_lower[i] and 
                  vol_confirmed and 
                  close[i] < ema200_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals