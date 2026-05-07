#!/usr/bin/env python3
# 6h_WeeklyPivot_RangeBreakout_1dTrend_Volume
# Hypothesis: Combines weekly pivot range (PP to R1/S1) with 1d trend filter and volume confirmation.
# In ranging markets (price between weekly S1 and R1), fade at weekly S1/R1 with trend filter.
# In trending markets (price breaks weekly R1 or S1), continue in breakout direction with volume.
# Uses weekly pivot for structure, 1d EMA50 for trend, and volume spike for confirmation.
# Designed for low frequency (15-25 trades/year) to minimize fee drag on 6h timeframe.
# Works in both bull and bear via trend filter and bidirectional logic.

name = "6h_WeeklyPivot_RangeBreakout_1dTrend_Volume"
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
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points (using prior week's data)
    # Pivot Point (PP) = (High + Low + Close) / 3
    # R1 = (2 * PP) - Low
    # S1 = (2 * PP) - High
    pp_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = (2 * pp_1w) - low_1w
    s1_1w = (2 * pp_1w) - high_1w
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly pivots and daily EMA to 6h timeframe
    pp_6h = align_htf_to_ltf(prices, df_1w, pp_1w)
    r1_6h = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_6h = align_htf_to_ltf(prices, df_1w, s1_1w)
    ema_50_1d_6h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike filter on 6h (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(pp_6h[i]) or np.isnan(r1_6h[i]) or np.isnan(s1_6h[i]) or 
            np.isnan(ema_50_1d_6h[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # Determine market regime based on weekly pivot
            # Ranging market: price between S1 and R1
            # Trending market: price breaks above R1 or below S1
            
            # Long conditions
            if close[i] > r1_6h[i]:  # Break above weekly R1 - bullish breakout
                if close[i] > ema_50_1d_6h[i] and volume_spike[i]:  # Trend and volume confirmation
                    signals[i] = 0.25
                    position = 1
                    bars_since_entry = 0
            elif close[i] < s1_6h[i] and close[i] > ema_50_1d_6h[i]:  # Pullback to S1 in uptrend
                if volume_spike[i]:  # Volume confirmation on pullback
                    signals[i] = 0.25
                    position = 1
                    bars_since_entry = 0
            
            # Short conditions
            elif close[i] < s1_6h[i]:  # Break below weekly S1 - bearish breakdown
                if close[i] < ema_50_1d_6h[i] and volume_spike[i]:  # Trend and volume confirmation
                    signals[i] = -0.25
                    position = -1
                    bars_since_entry = 0
            elif close[i] > r1_6h[i] and close[i] < ema_50_1d_6h[i]:  # Pullback to R1 in downtrend
                if volume_spike[i]:  # Volume confirmation on pullback
                    signals[i] = -0.25
                    position = -1
                    bars_since_entry = 0
        
        elif position == 1:
            # Long exit conditions
            if bars_since_entry >= 3:  # Minimum holding period
                # Exit if price breaks below weekly S1 or trend turns bearish
                if close[i] < s1_6h[i] or close[i] < ema_50_1d_6h[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            else:
                # Hold position for minimum period
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit conditions
            if bars_since_entry >= 3:  # Minimum holding period
                # Exit if price breaks above weekly R1 or trend turns bullish
                if close[i] > r1_6h[i] or close[i] > ema_50_1d_6h[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
            else:
                # Hold position for minimum period
                signals[i] = -0.25
    
    return signals