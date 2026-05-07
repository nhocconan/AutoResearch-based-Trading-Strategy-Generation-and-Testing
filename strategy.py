#!/usr/bin/env python3
"""
1d_Weekly_Pivot_Breakout_Trend_Filter_v2
Hypothesis: On 1d timeframe, buy when price breaks above weekly pivot R1 level with weekly trend filter (price > weekly EMA50); sell when breaks below weekly S1 level with weekly trend filter (price < weekly EMA50). Uses weekly EMA50 for trend direction to avoid counter-trend whipsaws. Weekly pivot levels calculated from prior week's OHLC. Designed to work in both bull and bear markets by using trend filter and mean-reversion levels. Target: 7-25 trades per year on 1d timeframe.
"""
name = "1d_Weekly_Pivot_Breakout_Trend_Filter_v2"
timeframe = "1d"
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
    
    # Get weekly data for pivot levels and EMA
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly pivot levels from previous week's OHLC
    # Standard pivot: P = (H + L + C) / 3
    # R1 = 2*P - L
    # S1 = 2*P - H
    prev_high = df_1w['high'].shift(1).values
    prev_low = df_1w['low'].shift(1).values
    prev_close = df_1w['close'].shift(1).values
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    weekly_r1 = 2 * pivot - prev_low
    weekly_s1 = 2 * pivot - prev_high
    
    # Align weekly pivot levels to daily timeframe
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # Weekly EMA50 for trend filter
    ema_50 = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Volume filter: current volume > 1.5 * 20-period average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    start_idx = max(50, 20)  # Ensure sufficient warmup for EMA and volume
    
    for i in range(start_idx, n):
        bars_since_entry += 1
        
        # Skip if any data is not ready
        if (np.isnan(weekly_r1_aligned[i]) or np.isnan(weekly_s1_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        if position == 0:
            # Minimum 5 days between trades to reduce frequency (1d timeframe)
            if bars_since_entry < 5:
                continue
                
            # Long: price breaks above weekly R1 + price above weekly EMA50 + volume filter
            if (close[i] > weekly_r1_aligned[i] and 
                close[i] > ema_50_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Short: price breaks below weekly S1 + price below weekly EMA50 + volume filter
            elif (close[i] < weekly_s1_aligned[i] and 
                  close[i] < ema_50_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        elif position != 0:
            # Exit: price returns to opposite weekly pivot level (S1 for long, R1 for short)
            if position == 1:
                if close[i] < weekly_s1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > weekly_r1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals