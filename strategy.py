#!/usr/bin/env python3
# 4h_Pivot_Point_Squeeze_Breakout_Trend_Filter
# Hypothesis: Combines daily pivot point support/resistance with Bollinger Band squeeze
# to identify low-volatility breakouts. Uses 1-day EMA50 as trend filter and volume
# confirmation to avoid false breakouts. Works in both bull and bear markets by
# following the daily trend direction. Designed for low trade frequency (20-40/year)
# to minimize fee drag.

name = "4h_Pivot_Point_Squeeze_Breakout_Trend_Filter"
timeframe = "4h"
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
    
    # 1d data for pivot points, trend filter, and Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points (standard formula)
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L
    # S1 = 2*P - H
    pivot = (high_1d + low_1d + close_1d) / 3
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    
    # Align pivot levels to 4h timeframe (wait for 1d bar to close)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 1-day EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Bollinger Bands (20, 2) on 1-day timeframe for squeeze detection
    sma_20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20_1d + 2 * std_20_1d
    lower_bb = sma_20_1d - 2 * std_20_1d
    bb_width = (upper_bb - lower_bb) / sma_20_1d  # Normalized width
    
    # Align Bollinger Band width to 4h timeframe
    bb_width_aligned = align_htf_to_ltf(prices, df_1d, bb_width)
    
    # Volume confirmation (6-period average = 1.5 days for 4h timeframe)
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p-1, len(arr)):
                res[i] = np.mean(arr[i-p+1:i+1])
        return res
    vol_ma = mean_arr(volume, 6)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for EMA and BB
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(ema_50_aligned[i]) or
            np.isnan(bb_width_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Bollinger Band squeeze condition: low volatility (< 5th percentile of recent width)
            # For simplicity, use fixed threshold - adjust based on empirical observation
            squeeze_condition = bb_width_aligned[i] < 0.02  # 2% width threshold
            
            # Long: price breaks above R1, above 1d EMA50, volume confirmation, during squeeze
            if (squeeze_condition and 
                close[i] > r1_aligned[i] and 
                close[i] > ema_50_aligned[i] and 
                volume[i] > 1.5 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1, below 1d EMA50, volume confirmation, during squeeze
            elif (squeeze_condition and 
                  close[i] < s1_aligned[i] and 
                  close[i] < ema_50_aligned[i] and 
                  volume[i] > 1.5 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price drops below pivot OR below 1d EMA50
            if close[i] < pivot_aligned[i] or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises above pivot OR above 1d EMA50
            if close[i] > pivot_aligned[i] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals