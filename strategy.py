#!/usr/bin/env python3
"""
6h_WeeklyPivot_R1_S1_Breakout_VolumeTrend
Hypothesis: On 6h timeframe, price breaks out from weekly pivot R1/S1 levels with volume confirmation.
Long when price > weekly R1 with volume > 1.5x average. Short when price < weekly S1 with volume > 1.5x average.
Weekly pivots provide structural support/resistance that works in both bull and bear markets.
Volume confirms institutional participation. Trend filter avoids counter-trend entries.
Target: 20-50 trades/year, holding period 2-5 days. Survives 2022 crash via trend alignment.
"""

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
    
    # === Weekly data for pivot calculation ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points (using prior week's OHLC)
    prev_high_1w = np.roll(high_1w, 1)
    prev_low_1w = np.roll(low_1w, 1)
    prev_close_1w = np.roll(close_1w, 1)
    
    # Handle first week
    prev_high_1w[0] = high_1w[0]
    prev_low_1w[0] = low_1w[0]
    prev_close_1w[0] = close_1w[0]
    
    # Weekly pivot calculation (standard floor pivot)
    pivot_1w = (prev_high_1w + prev_low_1w + prev_close_1w) / 3.0
    weekly_range = prev_high_1w - prev_low_1w
    
    # Weekly R1 and S1 levels
    weekly_r1 = 2 * pivot_1w - prev_low_1w
    weekly_s1 = 2 * pivot_1w - prev_high_1w
    
    # Align weekly levels to 6h timeframe
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # 6-day EMA for trend filter (avoid counter-trend trades)
    ema6 = pd.Series(close).ewm(span=6, min_periods=6, adjust=False).mean().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > 1.5 * vol_ma20
    
    signals = np.zeros(n)
    
    # Warmup covers EMA6, volume MA20, and weekly rollouts
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(weekly_r1_aligned[i]) or 
            np.isnan(weekly_s1_aligned[i]) or 
            np.isnan(ema6[i]) or 
            np.isnan(vol_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions: price breaks above weekly R1 with volume and uptrend
        if close[i] > weekly_r1_aligned[i] and vol_filter[i] and close[i] > ema6[i]:
            signals[i] = 0.25
        
        # Short conditions: price breaks below weekly S1 with volume and downtrend
        elif close[i] < weekly_s1_aligned[i] and vol_filter[i] and close[i] < ema6[i]:
            signals[i] = -0.25
        
        # Otherwise flat
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_WeeklyPivot_R1_S1_Breakout_VolumeTrend"
timeframe = "6h"
leverage = 1.0