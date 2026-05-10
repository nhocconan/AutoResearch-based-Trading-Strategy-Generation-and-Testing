#!/usr/bin/env python3
"""
1d_WeeklyTrend_RangeFilter_Volume
Hypothesis: On the daily timeframe, use weekly trend direction (via weekly close > weekly open) 
combined with daily range expansion (today's range > 1.5x average weekly range) and volume confirmation 
to capture trending moves while avoiding chop. Weekly trend filters noise, daily range/volume 
provides timely entry. Designed for low turnover (target 10-25 trades/year) to minimize fee drag 
and work in both bull and bear markets via trend-following logic.
"""

name = "1d_WeeklyTrend_RangeFilter_Volume"
timeframe = "1d"
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
    
    # Get weekly data for trend and range filters
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Weekly trend: bullish if weekly close > weekly open
    weekly_close = df_1w['close'].values
    weekly_open = df_1w['open'].values
    weekly_bullish = weekly_close > weekly_open
    
    # Weekly average true range for volatility filter
    # Calculate True Range for weekly data
    tr1 = df_1w['high'].values - df_1w['low'].values
    tr2 = np.abs(df_1w['high'].values - np.roll(weekly_close, 1))
    tr3 = np.abs(df_1w['low'].values - np.roll(weekly_close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Set first TR to high-low (no prior close)
    tr[0] = df_1w['high'].values[0] - df_1w['low'].values[0]
    # Average TR over 4 weeks for stability
    atr_4w = pd.Series(tr).rolling(window=4, min_periods=4).mean().values
    
    # Align weekly indicators to daily
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    atr_4w_aligned = align_htf_to_ltf(prices, df_1w, atr_4w)
    
    # Daily range (high - low)
    daily_range = high - low
    
    # Daily average volume (20-day)
    vol_avg_20d = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need weekly trend (4 weeks) and volume (20 days)
    start_idx = max(20, 4)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(weekly_bullish_aligned[i]) or 
            np.isnan(atr_4w_aligned[i]) or 
            np.isnan(vol_avg_20d[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Conditions
        weekly_trend_up = weekly_bullish_aligned[i] > 0.5  # bullish week
        range_expansion = daily_range[i] > (atr_4w_aligned[i] * 1.5)  # volatile day
        volume_confirm = volume[i] > (vol_avg_20d[i] * 1.5)  # high volume
        
        if position == 0:
            # Long entry: weekly uptrend + range expansion + volume
            if weekly_trend_up and range_expansion and volume_confirm:
                signals[i] = 0.25
                position = 1
        elif position == 1:
            # Long exit: weekly trend turns bearish OR volatility dries up
            if not weekly_trend_up or not range_expansion:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
    
    return signals