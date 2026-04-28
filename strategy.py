#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter and volume confirmation
# Long when price breaks above Donchian(20) high AND close > weekly pivot AND volume > 1.8x 30-bar avg
# Short when price breaks below Donchian(20) low AND close < weekly pivot AND volume > 1.8x 30-bar avg
# Exit when price retouches opposite Donchian level (low for longs, high for shorts)
# Uses discrete position sizing (0.25) to minimize fee churn. Target: 12-37 trades/year on 6h.
# Weekly pivot provides structural bias that works in both bull (breakouts with trend) and bear (fade at resistance).
# Volume confirmation ensures breakouts have conviction, reducing false signals in choppy markets.
# 6h timeframe balances trade frequency and noise reduction.

name = "6h_Donchian20_WeeklyPivot_Direction_VolumeFilter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for weekly pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:  # Need at least one weekly bar
        return np.zeros(n)
    
    # Calculate weekly pivot points from previous 1w OHLC
    # Standard pivot: P = (H + L + C) / 3
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    weekly_pivot = (high_1w + low_1w + close_1w) / 3.0
    
    # Align weekly pivot to 6h timeframe (use completed 1w bar's pivot)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Donchian(20) on primary timeframe (6h)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: >1.8x 30-bar average volume (moderate filter for 6h)
    volume_series = pd.Series(volume)
    volume_ma_30 = volume_series.rolling(window=30, min_periods=30).mean().values
    volume_confirm = volume > 1.8 * volume_ma_30
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback, 30)  # Need sufficient history for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(volume_ma_30[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        pivot = weekly_pivot_aligned[i]
        upper_channel = highest_high[i]
        lower_channel = lowest_low[i]
        curr_close = close[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when price breaks above Donchian high AND close > weekly pivot AND volume confirmation
            if curr_close > upper_channel and curr_close > pivot and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below Donchian low AND close < weekly pivot AND volume confirmation
            elif curr_close < lower_channel and curr_close < pivot and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when price retouches Donchian low
            if curr_close <= lower_channel:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when price retouches Donchian high
            if curr_close >= upper_channel:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals