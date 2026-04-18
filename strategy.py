#!/usr/bin/env python3
"""
12h Donchian Breakout with 1w Trend Filter and Volume Confirmation
Hypothesis: Price breaking weekly Donchian channels with 1-week trend alignment
and volume confirmation captures sustained moves in both bull and bear markets.
The 12h timeframe reduces trade frequency to minimize fee drag while the weekly
trend filter ensures we trade with the dominant higher timeframe momentum.
Volume confirmation filters out false breakouts. Target: 15-30 trades/year.
"""

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
    
    # Get weekly data for trend filter and Donchian channels (once before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly EMA40 for trend filter
    ema40_1w = pd.Series(df_1w['close'].values).ewm(span=40, adjust=False, min_periods=40).mean().values
    ema40_1w_aligned = align_htf_to_ltf(prices, df_1w, ema40_1w)
    
    # Weekly Donchian channels (20-period)
    high_20 = pd.Series(df_1w['high'].values).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(df_1w['low'].values).rolling(window=20, min_periods=20).min().values
    high_20_aligned = align_htf_to_ltf(prices, df_1w, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1w, low_20)
    
    # Volume filter: current volume > 1.8x 30-period volume average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_filter = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 45  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema40_1w_aligned[i]) or np.isnan(high_20_aligned[i]) or 
            np.isnan(low_20_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        trend = ema40_1w_aligned[i]
        upper = high_20_aligned[i]
        lower = low_20_aligned[i]
        vol_ok = vol_filter[i]
        
        if position == 0:
            # Long: price breaks above weekly Donchian upper with volume, in uptrend
            if price > upper and vol_ok and price > trend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly Donchian lower with volume, in downtrend
            elif price < lower and vol_ok and price < trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit if price returns to weekly Donchian middle or trend weakens
            mid = (upper + lower) / 2.0
            if price < mid or price < trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit if price returns to weekly Donchian middle or trend weakens
            mid = (upper + lower) / 2.0
            if price > mid or price > trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian_WeeklyTrend_Volume"
timeframe = "12h"
leverage = 1.0