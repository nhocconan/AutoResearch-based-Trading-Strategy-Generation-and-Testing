#!/usr/bin/env python3
# 6h_Donchian_Breakout_1dTrend_Volume
# Hypothesis: 6h Donchian breakout with 1d EMA trend filter and volume confirmation.
# Long when price breaks above 6h 20-period Donchian high, above 1d EMA50, and volume > 1.5x 20-period average.
# Short when price breaks below 6h 20-period Donchian low, below 1d EMA50, and volume > 1.5x 20-period average.
# Exit on opposite Donchian breakout or trend reversal.
# Designed for low frequency (15-30 trades/year) to avoid fee drag. Trend filter avoids false breakouts in chop.

name = "6h_Donchian_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 60-period volume average for confirmation
    vol_ma = np.zeros(n)
    vol_sum = 0.0
    vol_count = 0
    for i in range(n):
        vol_sum += volume[i]
        vol_count += 1
        if vol_count > 20:
            vol_sum -= volume[i - 20]
        else:
            vol_count = min(vol_count, 20)
        if vol_count == 20:
            vol_ma[i] = vol_sum / 20.0
    
    # Calculate Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(n):
        if i >= 19:  # 20 periods needed
            donchian_high[i] = np.max(high[i-19:i+1])
            donchian_low[i] = np.min(low[i-19:i+1])
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        multiplier = 2.0 / (50 + 1)
        ema_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_1d[i] = (close_1d[i] - ema_1d[i-1]) * multiplier + ema_1d[i-1]
    
    # Align 1d EMA to 6h timeframe
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Wait for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # LONG: Break above Donchian high + above 1d EMA50 + volume confirmation
            if close[i] > donchian_high[i] and close[i] > ema_1d_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # SHORT: Break below Donchian low + below 1d EMA50 + volume confirmation
            elif close[i] < donchian_low[i] and close[i] < ema_1d_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Break below Donchian low OR trend reversal (below EMA)
            if close[i] < donchian_low[i] or close[i] < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Break above Donchian high OR trend reversal (above EMA)
            if close[i] > donchian_high[i] or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals