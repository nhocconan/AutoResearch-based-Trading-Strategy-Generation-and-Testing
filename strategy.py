#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_1wPivotTrend_VolumeSpike
Hypothesis: 6h Donchian(20) breakout with weekly pivot trend filter and volume spike confirmation.
Long when price breaks above 20-bar high in weekly uptrend (close > weekly pivot) with volume > 2x average.
Short when price breaks below 20-bar low in weekly downtrend (close < weekly pivot) with volume > 2x average.
Exit when price re-enters 20-bar Donchian channel or weekly trend reverses.
Uses discrete position sizing (0.25) to minimize fee churn and target ~15-25 trades/year.
Designed to work in both bull and bear markets by aligning with weekly structure and requiring volume confirmation.
"""

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
    
    # Get weekly data for pivot trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot (standard: (H+L+C)/3)
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    
    # Align weekly pivot to 6h timeframe
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    
    # Calculate 6h Donchian channels (20-period)
    donchian_window = 20
    highest_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    lowest_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # Volume confirmation: volume > 2x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = max(50, donchian_window)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(pivot_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        weekly_pivot = pivot_1w_aligned[i]
        donchian_high = highest_high[i]
        donchian_low = lowest_low[i]
        
        if position == 0:
            # Wait for weekly trend alignment
            if close[i] > weekly_pivot:  # Weekly uptrend
                # Long: break above Donchian high with volume spike
                long_signal = (close[i] > donchian_high) and vol_spike[i]
            else:  # Weekly downtrend
                # Short: break below Donchian low with volume spike
                short_signal = (close[i] < donchian_low) and vol_spike[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price re-enters Donchian channel OR weekly trend turns down
            exit_signal = (close[i] < donchian_high and close[i] > donchian_low) or (close[i] < weekly_pivot)
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price re-enters Donchian channel OR weekly trend turns up
            exit_signal = (close[i] > donchian_low and close[i] < donchian_high) or (close[i] > weekly_pivot)
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Donchian20_Breakout_1wPivotTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0