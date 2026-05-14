#!/usr/bin/env python3
# 6h_weekly_volume_pivot_v2
# Hypothesis: 6h strategy using weekly pivot points with volume confirmation and weekly trend filter.
# Weekly trend filter (price above/below weekly 20 EMA) avoids counter-trend trades.
# Volume confirmation (>1.3x 20-period average) ensures breakout validity.
# Discrete sizing (0.0, ±0.25) minimizes fee churn. Target: 12-25 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_volume_pivot_v2"
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
    
    # 1w HTF data for weekly pivot points and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_w = df_1w['high'].values
    low_w = df_1w['low'].values
    close_w = df_1w['close'].values
    
    # Weekly pivot points (standard formula)
    pivot_point = (high_w + low_w + close_w) / 3.0
    r1 = 2 * pivot_point - low_w
    s1 = 2 * pivot_point - high_w
    r2 = pivot_point + (high_w - low_w)
    s2 = pivot_point - (high_w - low_w)
    
    # Weekly 20 EMA for trend filter
    ema_20_1w = pd.Series(close_w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align weekly data to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1w, pivot_point)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    ema_20_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(ema_20_aligned[i]) or
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirmed = volume[i] > 1.3 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: price reaches R2 or volume dries up
            if close[i] >= r2_aligned[i] or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches S2 or volume dries up
            if close[i] <= s2_aligned[i] or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if volume_confirmed:
                # Long breakout: price breaks above R1 with volume AND weekly trend filter (price > weekly 20 EMA)
                if close[i] > r1_aligned[i] and close[i] > ema_20_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short breakdown: price breaks below S1 with volume AND weekly trend filter (price < weekly 20 EMA)
                elif close[i] < s1_aligned[i] and close[i] < ema_20_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals