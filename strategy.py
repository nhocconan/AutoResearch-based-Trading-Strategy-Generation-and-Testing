#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with 1-week trend filter (EMA34) and 6h Ichimoku cloud twist (conversion/base line cross) with volume confirmation.
# Enters only during 08-20 UTC session. Uses Ichimoku conversion line (9) and base line (26) crossover with cloud filter from daily timeframe.
# Trend-following in bull markets, avoids false signals in bear/chop via weekly EMA34 filter and volume spike requirement.
# Target: 50-150 total trades over 4 years = 12-37/year.
name = "6h_1w_EMA34_Ichimoku9_26_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    # Get 1w data for EMA34 trend (called ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Get 1d data for Ichimoku cloud (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    # Ichimoku components: conversion line (9), base line (26)
    high_9_1d = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    low_9_1d = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    high_26_1d = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    low_26_1d = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    conversion_line_1d = (high_9_1d + low_9_1d) / 2
    base_line_1d = (high_26_1d + low_26_1d) / 2
    # Leading Span A and B for cloud
    span_a_1d = (conversion_line_1d + base_line_1d) / 2
    span_b_1d = (pd.Series(high_26_1d).rolling(window=26, min_periods=26).max().values + 
                 pd.Series(low_26_1d).rolling(window=26, min_periods=26).min().values) / 2
    # Align Ichimoku components to 6h
    conversion_line_1d_aligned = align_htf_to_ltf(prices, df_1d, conversion_line_1d)
    base_line_1d_aligned = align_htf_to_ltf(prices, df_1d, base_line_1d)
    span_a_1d_aligned = align_htf_to_ltf(prices, df_1d, span_a_1d)
    span_b_1d_aligned = align_htf_to_ltf(prices, df_1d, span_b_1d)
    
    # Volume filter: volume > 2.0 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(conversion_line_1d_aligned[i]) or 
            np.isnan(base_line_1d_aligned[i]) or np.isnan(span_a_1d_aligned[i]) or 
            np.isnan(span_b_1d_aligned[i]) or np.isnan(volume_ma[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above 1w EMA34 AND conversion > base AND price above cloud (span A and B) with volume
            if (close[i] > ema_34_1w_aligned[i] and 
                conversion_line_1d_aligned[i] > base_line_1d_aligned[i] and
                close[i] > max(span_a_1d_aligned[i], span_b_1d_aligned[i]) and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below 1w EMA34 AND conversion < base AND price below cloud (span A and B) with volume
            elif (close[i] < ema_34_1w_aligned[i] and 
                  conversion_line_1d_aligned[i] < base_line_1d_aligned[i] and
                  close[i] < min(span_a_1d_aligned[i], span_b_1d_aligned[i]) and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below 1w EMA34 or conversion < base or price below cloud
            if (close[i] < ema_34_1w_aligned[i] or 
                conversion_line_1d_aligned[i] < base_line_1d_aligned[i] or
                close[i] < min(span_a_1d_aligned[i], span_b_1d_aligned[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above 1w EMA34 or conversion > base or price above cloud
            if (close[i] > ema_34_1w_aligned[i] or 
                conversion_line_1d_aligned[i] > base_line_1d_aligned[i] or
                close[i] > max(span_a_1d_aligned[i], span_b_1d_aligned[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals