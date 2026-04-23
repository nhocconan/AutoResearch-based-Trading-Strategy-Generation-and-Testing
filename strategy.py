#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter and 1d volume confirmation.
Uses 1w pivot points (based on prior week's OHLC) to establish long-term bias.
Only takes longs when price > weekly pivot, shorts when price < weekly pivot.
Adds 1d volume spike filter (1.5x 20-period MA) to confirm institutional interest.
Target: 12-30 trades/year per symbol (50-120 total over 4 years).
Uses discrete position sizing (0.25) to minimize fee churn.
Works in bull markets via breakout continuation and bear markets via mean reversion at extremes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate weekly pivot points (prior week's OHLC)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Weekly pivot: (Prior week HIGH + LOW + CLOSE) / 3
    weekly_pivot = (df_1w['high'].values + df_1w['low'].values + df_1w['close'].values) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Calculate 1d volume MA for confirmation filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get 1d volume and compute 20-period MA
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate Donchian channels (20-period) on 6h data
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 20)  # need Donchian20, volume MA20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(weekly_pivot_aligned[i]) or 
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Donchian breakout conditions
        breakout_up = close[i] > highest_20[i-1]  # Break above prior 20-period high
        breakout_down = close[i] < lowest_20[i-1]  # Break below prior 20-period low
        
        # Weekly pivot bias: long above pivot, short below pivot
        bias_long = close[i] > weekly_pivot_aligned[i]
        bias_short = close[i] < weekly_pivot_aligned[i]
        
        # Volume confirmation: 1d volume > 1.5x 20-period MA
        vol_filter = volume[i] > 1.5 * vol_ma_20_1d_aligned[i]
        
        if position == 0:
            # Long: Donchian breakout up AND bullish bias AND volume confirmation
            if breakout_up and bias_long and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakout down AND bearish bias AND volume confirmation
            elif breakout_down and bias_short and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: opposite Donchian breakout or loss of bias
            exit_signal = False
            if position == 1:
                # Exit long on Donchian breakdown or bearish bias
                exit_signal = breakout_down or not bias_long
            elif position == -1:
                # Exit short on Donchian breakout or bullish bias
                exit_signal = breakout_up or not bias_short
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Donchian20_Breakout_WeeklyPivot_VolumeFilter"
timeframe = "6h"
leverage = 1.0