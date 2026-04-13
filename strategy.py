#!/usr/bin/env python3
"""
6h_1W_1D_Camarilla_Pivot_Direction_Filter_v1
Hypothesis: Use weekly Camarilla pivot direction (based on weekly close vs Pivot) as trend filter for 6h timeframe.
Enter long when 6h price breaks above daily Camarilla H3 level with volume > 2x 20-period average AND weekly trend is up (weekly close > weekly Pivot).
Enter short when 6h price breaks below daily Camarilla L3 level with volume > 2x 20-period average AND weekly trend is down (weekly close < weekly Pivot).
This combines daily breakout precision with weekly trend filter to avoid counter-trend trades, reducing false signals in both bull and bear markets.
Target: 50-150 trades over 4 years (12-37/year) to minimize fee drag.
"""

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
    
    # Volume confirmation: current volume > 2x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 2.0)
    
    # Daily data for Camarilla levels (breakout levels)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    prev_high_1d = df_1d['high'].values
    prev_low_1d = df_1d['low'].values
    prev_close_1d = df_1d['close'].values
    
    # Calculate daily Camarilla levels (H3/L3 for entries)
    daily_range = prev_high_1d - prev_low_1d
    camarilla_h3_1d = prev_close_1d + 1.1 * daily_range / 4  # H3 level
    camarilla_l3_1d = prev_close_1d - 1.1 * daily_range / 4  # L3 level
    
    # Align daily levels to 6h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3_1d)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3_1d)
    
    # Weekly data for trend filter (Pivot and close)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    prev_high_1w = df_1w['high'].values
    prev_low_1w = df_1w['low'].values
    prev_close_1w = df_1w['close'].values
    
    # Calculate weekly Pivot point (standard formula)
    weekly_pivot = (prev_high_1w + prev_low_1w + prev_close_1w) / 3.0
    
    # Weekly trend: up if close > pivot, down if close < pivot
    weekly_trend_up = prev_close_1w > weekly_pivot
    weekly_trend_down = prev_close_1w < weekly_pivot
    
    # Align weekly trend to 6h timeframe
    weekly_trend_up_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_up.astype(float))
    weekly_trend_down_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_down.astype(float))
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    bars_since_entry = 0  # Track holding period to prevent whipsaw
    
    for i in range(50, n):  # warmup period
        # Skip if any required data is not ready
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(volume_expansion[i]) or np.isnan(weekly_trend_up_aligned[i]) or 
            np.isnan(weekly_trend_down_aligned[i])):
            signals[i] = 0.0
            bars_since_entry = 0
            continue
        
        bars_since_entry += 1
        
        # Long conditions: break above daily H3 + volume expansion + weekly uptrend
        long_signal = (close[i] > camarilla_h3_aligned[i] and 
                      volume_expansion[i] and 
                      weekly_trend_up_aligned[i] > 0.5)
        
        # Short conditions: break below daily L3 + volume expansion + weekly downtrend
        short_signal = (close[i] < camarilla_l3_aligned[i] and 
                       volume_expansion[i] and 
                       weekly_trend_down_aligned[i] > 0.5)
        
        # Exit conditions: minimum 6 bars held (36 hours) OR opposite signal
        if position == 1 and (bars_since_entry >= 6 or short_signal):
            position = -1 if short_signal else 0
            signals[i] = -position_size if short_signal else 0.0
            bars_since_entry = 0
        elif position == -1 and (bars_since_entry >= 6 or long_signal):
            position = 1 if long_signal else 0
            signals[i] = position_size if long_signal else 0.0
            bars_since_entry = 0
        elif position == 0:
            if long_signal:
                position = 1
                signals[i] = position_size
                bars_since_entry = 0
            elif short_signal:
                position = -1
                signals[i] = -position_size
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "6h_1W_1D_Camarilla_Pivot_Direction_Filter_v1"
timeframe = "6h"
leverage = 1.0