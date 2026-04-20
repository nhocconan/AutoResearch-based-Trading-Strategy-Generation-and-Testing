#!/usr/bin/env python3
"""
4h_Choppiness_Breakout_With_Volume_Confirmation
Hypothesis: Trade breakouts of Donchian channels (20-period) in trending markets only (Choppiness Index < 38.2) with volume confirmation (>1.5x 20-period average). 
Long when price breaks above upper Donchian band in trend, short when breaks below lower band. Uses 4h timeframe to capture medium-term moves while reducing noise.
Choppiness Index filter avoids whipsaws in ranging markets. Volume confirmation reduces false breakouts.
Works in bull/bear: Trend filter ensures trades align with market direction, volume filter improves signal quality.
Target: 100-200 total trades over 4 years (25-50/year) with position size 0.25.
"""

name = "4h_Choppiness_Breakout_With_Volume_Confirmation"
timeframe = "4h"
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
    
    # Calculate Donchian channels (20-period)
    def rolling_max(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            result[i] = np.max(arr[i-window+1:i+1])
        return result
    
    def rolling_min(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            result[i] = np.min(arr[i-window+1:i+1])
        return result
    
    upper_donchian = rolling_max(high, 20)
    lower_donchian = rolling_min(low, 20)
    
    # Calculate Choppiness Index (14-period)
    def true_range(high, low, close):
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = high[0] - low[0]  # First period
        return tr
    
    def atr(high, low, close, period):
        tr = true_range(high, low, close)
        result = np.full_like(tr, np.nan)
        if len(tr) >= period:
            result[period-1] = np.mean(tr[:period])
            for i in range(period, len(tr)):
                result[i] = (result[i-1] * (period-1) + tr[i]) / period
        return result
    
    tr = true_range(high, low, close)
    atr14 = atr(high, low, close, 14)
    
    def highest(high, period):
        result = np.full_like(high, np.nan)
        for i in range(period-1, len(high)):
            result[i] = np.max(high[i-period+1:i+1])
        return result
    
    def lowest(low, period):
        result = np.full_like(low, np.nan)
        for i in range(period-1, len(low)):
            result[i] = np.min(low[i-period+1:i+1])
        return result
    
    highest_high = highest(high, 14)
    lowest_low = lowest(low, 14)
    
    # Choppiness Index: 100 * log10(sum(TR14) / (ATR14 * 14)) / log10(14)
    sum_tr14 = np.full_like(tr, np.nan)
    for i in range(13, len(tr)):
        sum_tr14[i] = np.sum(tr[i-13:i+1])
    
    chop = np.full_like(tr, np.nan)
    for i in range(13, len(tr)):
        if atr14[i] > 0 and sum_tr14[i] > 0:
            chop[i] = 100 * np.log10(sum_tr14[i] / (atr14[i] * 14)) / np.log10(14)
        else:
            chop[i] = 50  # Neutral when undefined
    
    # Calculate volume filter (volume > 1.5x 20-period average)
    vol_ma20 = np.full_like(volume, np.nan)
    for i in range(20, len(volume)):
        vol_ma20[i] = np.mean(volume[i-20:i])
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure indicators are ready (20 for Donchian + buffer)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_donchian[i]) or np.isnan(lower_donchian[i]) or 
            np.isnan(chop[i]) or np.isnan(volume[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above upper Donchian band in trending market (CHOP < 38.2) with volume confirmation
            if close[i] > upper_donchian[i] and chop[i] < 38.2 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian band in trending market (CHOP < 38.2) with volume confirmation
            elif close[i] < lower_donchian[i] and chop[i] < 38.2 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below lower Donchian band OR choppiness increases (CHOP > 61.8)
            if close[i] < lower_donchian[i] or chop[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above upper Donchian band OR choppiness increases (CHOP > 61.8)
            if close[i] > upper_donchian[i] or chop[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals