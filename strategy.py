#!/usr/bin/env python3
# 6h_WeeklyPivot_DonchianBreakout
# Hypothesis: 6-hour Donchian(20) breakout with weekly pivot bias and volume confirmation
# Weekly pivot provides long-term bias (from weekly high/low/close) to filter breakout direction
# Works in bull markets via breakout momentum with upward bias and in bear via short breakdowns with downward bias
# Volume filter reduces false breakouts. Target: 50-150 total trades over 4 years with position size 0.25

name = "6h_WeeklyPivot_DonchianBreakout"
timeframe = "6h"
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
    
    # Load weekly data ONCE for pivot bias
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week's OHLC)
    # Pivot = (H + L + C)/3
    # Support 1 = 2*P - H
    # Resistance 1 = 2*P - L
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    r1 = 2 * pivot - weekly_high
    s1 = 2 * pivot - weekly_low
    
    # Bias: above pivot = bullish bias, below pivot = bearish bias
    weekly_bias = pivot  # Using pivot as bias reference
    
    # Align weekly bias to 6h timeframe (wait for weekly close)
    weekly_bias_aligned = align_htf_to_ltf(prices, df_1w, weekly_bias)
    
    # Donchian channels (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume ratio: current volume / 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need 20 periods for Donchian
    
    for i in range(start_idx, n):
        if (np.isnan(weekly_bias_aligned[i]) or np.isnan(high_max[i]) or 
            np.isnan(low_min[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Breakout conditions
        breakout_up = close[i] > high_max[i-1]  # Break above previous period's high
        breakout_down = close[i] < low_min[i-1]  # Break below previous period's low
        
        # Volume confirmation: volume > 1.5x average
        volume_confirm = vol_ratio[i] > 1.5
        
        # Weekly pivot bias: price above/below weekly pivot
        price_above_pivot = close[i] > weekly_bias_aligned[i]
        price_below_pivot = close[i] < weekly_bias_aligned[i]
        
        if position == 0:
            # Long: upward breakout + volume + price above weekly pivot
            if breakout_up and volume_confirm and price_above_pivot:
                signals[i] = 0.25
                position = 1
            # Short: downward breakout + volume + price below weekly pivot
            elif breakout_down and volume_confirm and price_below_pivot:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price breaks below Donchian low or crosses below weekly pivot
            if close[i] < low_min[i-1] or not price_above_pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks above Donchian high or crosses above weekly pivot
            if close[i] > high_max[i-1] or not price_below_pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals