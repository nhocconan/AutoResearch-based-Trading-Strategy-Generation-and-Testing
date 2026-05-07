#!/usr/bin/env python3
# 6h_1dSupertrend_1dTrend_HTFTrend
# Uses daily Supertrend for trend direction and weekly Supertrend for higher timeframe trend filter.
# Enters long when both daily and weekly Supertrend are bullish and price pulls back to daily Supertrend in an uptrend.
# Enters short when both are bearish and price bounces off daily Supertrend in a downtrend.
# Weekly Supertrend ensures we only trade with the dominant long-term trend, reducing whipsaws in sideways markets.
# Designed for 6h timeframe to capture medium-term trends with high conviction entries.

name = "6h_1dSupertrend_1dTrend_HTFTrend"
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
    
    # Get daily data for Supertrend calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get weekly data for higher timeframe trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate daily Supertrend (10, 3.0)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    atr_period = 10
    multiplier = 3.0
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.max([high_1d[0] - low_1d[0], np.abs(high_1d[0] - close_1d[0]), np.abs(low_1d[0] - close_1d[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR using Wilder's smoothing (equivalent to RMA)
    atr = np.zeros_like(close_1d)
    atr[atr_period] = np.mean(tr[:atr_period+1])
    for i in range(atr_period+1, len(atr)):
        atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # Supertrend calculation
    upper_band = (high_1d + low_1d) / 2 + multiplier * atr
    lower_band = (high_1d + low_1d) / 2 - multiplier * atr
    
    supertrend = np.zeros_like(close_1d)
    direction = np.ones_like(close_1d)  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = upper_band[0]
    direction[0] = 1
    
    for i in range(1, len(close_1d)):
        if close_1d[i] > supertrend[i-1]:
            direction[i] = 1
        elif close_1d[i] < supertrend[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
        
        if direction[i] == 1 and direction[i-1] == -1:
            supertrend[i] = lower_band[i]
        elif direction[i] == -1 and direction[i-1] == 1:
            supertrend[i] = upper_band[i]
        elif direction[i] == 1:
            supertrend[i] = max(lower_band[i], supertrend[i-1])
        else:
            supertrend[i] = min(upper_band[i], supertrend[i-1])
    
    # Calculate weekly Supertrend (10, 3.0) for trend filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    tr1_w = high_1w[1:] - low_1w[1:]
    tr2_w = np.abs(high_1w[1:] - close_1w[:-1])
    tr3_w = np.abs(low_1w[1:] - close_1w[:-1])
    tr_w = np.concatenate([[np.max([high_1w[0] - low_1w[0], np.abs(high_1w[0] - close_1w[0]), np.abs(low_1w[0] - close_1w[0])])], np.maximum(tr1_w, np.maximum(tr2_w, tr3_w))])
    
    atr_w = np.zeros_like(close_1w)
    atr_w[atr_period] = np.mean(tr_w[:atr_period+1])
    for i in range(atr_period+1, len(atr_w)):
        atr_w[i] = (atr_w[i-1] * (atr_period-1) + tr_w[i]) / atr_period
    
    upper_band_w = (high_1w + low_1w) / 2 + multiplier * atr_w
    lower_band_w = (high_1w + low_1w) / 2 - multiplier * atr_w
    
    supertrend_w = np.zeros_like(close_1w)
    direction_w = np.ones_like(close_1w)
    
    supertrend_w[0] = upper_band_w[0]
    direction_w[0] = 1
    
    for i in range(1, len(close_1w)):
        if close_1w[i] > supertrend_w[i-1]:
            direction_w[i] = 1
        elif close_1w[i] < supertrend_w[i-1]:
            direction_w[i] = -1
        else:
            direction_w[i] = direction_w[i-1]
        
        if direction_w[i] == 1 and direction_w[i-1] == -1:
            supertrend_w[i] = lower_band_w[i]
        elif direction_w[i] == -1 and direction_w[i-1] == 1:
            supertrend_w[i] = upper_band_w[i]
        elif direction_w[i] == 1:
            supertrend_w[i] = max(lower_band_w[i], supertrend_w[i-1])
        else:
            supertrend_w[i] = min(upper_band_w[i], supertrend_w[i-1])
    
    # Align daily and weekly Supertrend to 6h timeframe
    supertrend_1d = align_htf_to_ltf(prices, df_1d, supertrend)
    direction_1d = align_htf_to_ltf(prices, df_1d, direction)
    supertrend_1w = align_htf_to_ltf(prices, df_1w, supertrend_w)
    direction_1w = align_htf_to_ltf(prices, df_1w, direction_w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0  # Track holding period
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(supertrend_1d[i]) or np.isnan(direction_1d[i]) or 
            np.isnan(supertrend_1w[i]) or np.isnan(direction_1w[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # Long: both daily and weekly uptrend, price pulls back to touch/slightly below daily Supertrend
            if direction_1d[i] == 1 and direction_1w[i] == 1 and close[i] <= supertrend_1d[i] * 1.001:
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Short: both daily and weekly downtrend, price bounces off to touch/slightly above daily Supertrend
            elif direction_1d[i] == -1 and direction_1w[i] == -1 and close[i] >= supertrend_1d[i] * 0.999:
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        elif position == 1:
            # Exit: daily trend turns down or price moves significantly above Supertrend (take profit)
            if bars_since_entry >= 2 and (direction_1d[i] == -1 or close[i] > supertrend_1d[i] * 1.02):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: daily trend turns up or price moves significantly below Supertrend (take profit)
            if bars_since_entry >= 2 and (direction_1d[i] == 1 or close[i] < supertrend_1d[i] * 0.98):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
    
    return signals