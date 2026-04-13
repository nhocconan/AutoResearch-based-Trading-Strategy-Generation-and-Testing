#!/usr/bin/env python3
"""
6h_1D_1W_Camarilla_Pivot_Direction_Volume_Filter_v1
Hypothesis: Use weekly pivot direction as trend filter and daily Camarilla levels for entry signals on 6h timeframe. Enter long when price breaks above daily H3/H4 with volume > 1.5x 20-period average and weekly trend is bullish (close > weekly pivot). Enter short when price breaks below daily L3/L4 with volume confirmation and weekly trend bearish (close < weekly pivot). Weekly pivot provides higher timeframe trend bias to avoid counter-trend trades, while daily Camarilla levels offer precise entry/exit levels. Volume filter ensures breakouts have conviction. Designed for 60-100 trades over 4 years (15-25/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 1.5)
    
    # Daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    prev_high_1d = df_1d['high'].values
    prev_low_1d = df_1d['low'].values
    prev_close_1d = df_1d['close'].values
    
    # Calculate daily Camarilla levels (H3, L3, H4, L4)
    daily_range = prev_high_1d - prev_low_1d
    camarilla_h3_1d = prev_close_1d + 1.1 * daily_range / 4
    camarilla_l3_1d = prev_close_1d - 1.1 * daily_range / 4
    camarilla_h4_1d = prev_close_1d + 1.1 * daily_range / 2
    camarilla_l4_1d = prev_close_1d - 1.1 * daily_range / 2
    
    # Align daily levels to 6h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3_1d)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3_1d)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4_1d)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4_1d)
    
    # Weekly data for trend filter (pivot point)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    prev_high_1w = df_1w['high'].values
    prev_low_1w = df_1w['low'].values
    prev_close_1w = df_1w['close'].values
    
    # Weekly pivot point = (H + L + C) / 3
    weekly_pivot_1w = (prev_high_1w + prev_low_1w + prev_close_1w) / 3
    
    # Align weekly pivot to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(30, n):  # warmup period
        # Skip if any required data is not ready
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Determine weekly trend: bullish if close > weekly pivot, bearish if close < weekly pivot
        weekly_bullish = close[i] > weekly_pivot_aligned[i]
        weekly_bearish = close[i] < weekly_pivot_aligned[i]
        
        # Long signal: break above daily H3/H4 with volume expansion and weekly bullish trend
        long_signal = (volume_expansion[i] and weekly_bullish and
                      (close[i] > camarilla_h3_aligned[i] or close[i] > camarilla_h4_aligned[i]))
        
        # Short signal: break below daily L3/L4 with volume expansion and weekly bearish trend
        short_signal = (volume_expansion[i] and weekly_bearish and
                       (close[i] < camarilla_l3_aligned[i] or close[i] < camarilla_l4_aligned[i]))
        
        # Generate signals
        if long_signal and position != 1:
            position = 1
            signals[i] = position_size
        elif short_signal and position != -1:
            position = -1
            signals[i] = -position_size
        elif position == 1 and (close[i] < camarilla_l3_aligned[i]):  # Exit long if price breaks below L3
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > camarilla_h3_aligned[i]):  # Exit short if price breaks above H3
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "6h_1D_1W_Camarilla_Pivot_Direction_Volume_Filter_v1"
timeframe = "6h"
leverage = 1.0