#!/usr/bin/env python3
"""
6h Weekly Pivot + Volume Surge + Trend Filter
Hypothesis: Weekly pivots act as strong support/resistance in ranging markets. 
When price breaks above/below weekly pivot with volume surge and aligns with daily trend,
it signals institutional participation. Weekly timeframe reduces noise, volume surge 
confirms legitimacy, daily trend filter avoids counter-trend trades. Works in bull/bear
as it trades breakouts in both directions with strict filters.
"""

name = "6h_WeeklyPivot_VolumeTrend"
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
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points (standard: P = (H+L+C)/3, R1 = 2P-L, S1 = 2P-H)
    pivot_1w = np.full_like(close_1w, np.nan)
    r1_1w = np.full_like(close_1w, np.nan)
    s1_1w = np.full_like(close_1w, np.nan)
    
    for i in range(len(df_1w)):
        if not (np.isnan(high_1w[i]) or np.isnan(low_1w[i]) or np.isnan(close_1w[i])):
            pivot_1w[i] = (high_1w[i] + low_1w[i] + close_1w[i]) / 3.0
            r1_1w[i] = 2 * pivot_1w[i] - low_1w[i]
            s1_1w[i] = 2 * pivot_1w[i] - high_1w[i]
    
    # Align weekly pivot levels to 6h timeframe (wait for weekly bar close)
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 20-period EMA on daily for trend filter
    ema20_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 20:
        ema20_1d[19] = np.mean(close_1d[0:20])
        for i in range(20, len(close_1d)):
            ema20_1d[i] = (close_1d[i] * 2 + ema20_1d[i-1] * 18) / 20
    
    # Align daily EMA20 to 6h timeframe
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # Volume filter: current volume vs 20-period average
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid_vol = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid_vol] = volume[valid_vol] / vol_ma[valid_vol]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Need weekly pivot and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(pivot_1w_aligned[i]) or np.isnan(r1_1w_aligned[i]) or 
            np.isnan(s1_1w_aligned[i]) or np.isnan(ema20_1d_aligned[i]) or
            np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine conditions
        price_above_pivot = close[i] > pivot_1w_aligned[i]
        price_below_pivot = close[i] < pivot_1w_aligned[i]
        price_above_r1 = close[i] > r1_1w_aligned[i]
        price_below_s1 = close[i] < s1_1w_aligned[i]
        trend_up = close[i] > ema20_1d_aligned[i]
        volume_surge = volume_ratio[i] > 2.0
        
        if position == 0:
            # Enter long: Price above weekly pivot + volume surge + uptrend
            if price_above_pivot and volume_surge and trend_up:
                signals[i] = 0.25
                position = 1
            # Enter short: Price below weekly pivot + volume surge + downtrend
            elif price_below_pivot and volume_surge and not trend_up:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price breaks below pivot OR trend reverses
            if price_below_pivot or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price breaks above pivot OR trend reverses
            if price_above_pivot or trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals