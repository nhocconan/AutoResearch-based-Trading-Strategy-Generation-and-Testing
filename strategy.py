#!/usr/bin/env python3
"""
6h_Weekly_Pivot_1dTrend_Volume
Hypothesis: Weekly (Monday) pivot levels derived from prior week's OHLC act as key support/resistance in both bull and bear markets. Price breaking above weekly R1 or below S1 with 1d EMA34 trend alignment and volume confirmation captures momentum moves. Weekly pivot provides structural context, 1d trend filters noise, and volume confirms participation. Designed for low trade frequency on 6h timeframe to minimize fee drag.
"""
name = "6h_Weekly_Pivot_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot levels
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Weekly pivot points (using prior week's OHLC)
    high_w = df_weekly['high'].values
    low_w = df_weekly['low'].values
    close_w = df_weekly['close'].values
    
    pivot_w = (high_w + low_w + close_w) / 3.0
    r1_w = 2 * pivot_w - low_w
    s1_w = 2 * pivot_w - high_w
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align to 6h timeframe
    r1_w_aligned = align_htf_to_ltf(prices, df_weekly, r1_w)
    s1_w_aligned = align_htf_to_ltf(prices, df_weekly, s1_w)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filter: current volume > 1.3 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    start_idx = max(100, 20)
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(r1_w_aligned[i]) or np.isnan(s1_w_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # Long: price breaks above weekly R1 + 1d uptrend + volume
            if close[i] > r1_w_aligned[i] and close[i] > ema_34_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Short: price breaks below weekly S1 + 1d downtrend + volume
            elif close[i] < s1_w_aligned[i] and close[i] < ema_34_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        elif position != 0:
            # Minimum holding period of 4 bars to reduce trade frequency
            if bars_since_entry < 4:
                signals[i] = 0.25 if position == 1 else -0.25
                continue
            
            # Exit: price crosses back through the opposite S1/R1 level
            if position == 1:
                if close[i] < s1_w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > r1_w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals