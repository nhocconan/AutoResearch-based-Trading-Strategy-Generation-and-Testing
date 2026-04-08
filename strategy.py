#!/usr/bin/env python3
# 6h_weekly_pivot_trend_follow_v1
# Hypothesis: Use weekly pivot points (from 1w data) to determine long-term trend direction, then take 6-hour breakouts in the direction of the weekly trend with volume confirmation. Weekly pivots provide robust support/resistance that work in both bull and bear markets. Breakouts in the direction of the weekly trend capture momentum while avoiding counter-trend trades. Volume confirmation ensures breakouts are genuine. Target: 15-30 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_pivot_trend_follow_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points: P = (H+L+C)/3, R1 = 2P-L, S1 = 2P-H, R2 = P+(H-L), S2 = P-(H-L)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot = (high_1w + low_1w + close_1w) / 3.0
    r1 = 2 * pivot - low_1w
    s1 = 2 * pivot - high_1w
    r2 = pivot + (high_1w - low_1w)
    s2 = pivot - (high_1w - low_1w)
    
    # Trend determination: price above pivot = uptrend, below = downtrend
    weekly_trend = np.where(close_1w > pivot, 1, -1)  # 1=uptrend, -1=downtrend
    
    # Align weekly data to 6h
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    weekly_trend_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend)
    
    # 6h Donchian breakout (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume filter: 6h volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = max(lookback, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(pivot_aligned[i]) or np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or 
            np.isnan(weekly_trend_aligned[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price < weekly pivot OR price < 6h low (breakdown)
            if (close[i] < pivot_aligned[i]) or (close[i] < lowest_low[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price > weekly pivot OR price > 6h high (breakout)
            if (close[i] > pivot_aligned[i]) or (close[i] > highest_high[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price > 6h high (breakout) + weekly uptrend + volume
            if (close[i] > highest_high[i]) and (weekly_trend_aligned[i] == 1) and volume_filter[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: price < 6h low (breakdown) + weekly downtrend + volume
            elif (close[i] < lowest_low[i]) and (weekly_trend_aligned[i] == -1) and volume_filter[i]:
                position = -1
                signals[i] = -0.25
    
    return signals