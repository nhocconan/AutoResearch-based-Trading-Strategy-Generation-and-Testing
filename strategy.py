#!/usr/bin/env python3
# 6h_1d_WeeklyPivot_Breakout_VolumeTrend
# Hypothesis: Use weekly pivot levels (R1/S1) from 1d aggregated weekly data with 6h breakout and volume confirmation.
# Weekly pivot provides stronger support/resistance than daily, reducing false breakouts.
# Volume spike (1.5x 20-period average) confirms institutional interest.
# Trend filter: 6h EMA50 ensures trades align with intermediate trend.
# Target: 15-30 trades/year per symbol for low fee attrition and high edge.
# Works in bull/bear: breakouts capture momentum; weekly pivot adapts to longer-term structure.

name = "6h_1d_WeeklyPivot_Breakout_VolumeTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Aggregate 1d to weekly: use last 5 trading days (Mon-Fri)
    # Calculate weekly high, low, close from rolling 5-day window
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Weekly OHLC using 5-day rolling window (aligns with weekly candle close on Friday)
    # We'll calculate weekly pivot based on prior week's data to avoid look-ahead
    weekly_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().shift(5).values
    weekly_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().shift(5).values
    weekly_close = pd.Series(close_1d).rolling(window=5, min_periods=5).last().shift(5).values
    
    # Typical price for weekly pivot calculation
    weekly_typical = (weekly_high + weekly_low + weekly_close) / 3
    
    # Weekly pivot point and range
    weekly_pivot = weekly_typical
    weekly_range = weekly_high - weekly_low
    
    # Weekly Camarilla-like levels (using standard 1.1 multiplier for R1/S1)
    weekly_r1 = weekly_close + (weekly_range * 1.1 / 12)
    weekly_s1 = weekly_close - (weekly_range * 1.1 / 12)
    
    # Align weekly levels to 6h timeframe
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1d, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1d, weekly_s1)
    
    # Calculate 6h EMA50 for trend filter
    close_series = pd.Series(close)
    ema50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume average for spike detection (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(weekly_r1_aligned[i]) or np.isnan(weekly_s1_aligned[i]) or 
            np.isnan(ema50[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long breakout: price breaks above weekly R1 with volume spike and uptrend
            if (close[i] > weekly_r1_aligned[i] and 
                volume[i] > 1.5 * volume_ma[i] and 
                close[i] > ema50[i]):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below weekly S1 with volume spike and downtrend
            elif (close[i] < weekly_s1_aligned[i] and 
                  volume[i] > 1.5 * volume_ma[i] and 
                  close[i] < ema50[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below weekly S1 or trend reverses
            if close[i] < weekly_s1_aligned[i] or close[i] < ema50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above weekly R1 or trend reverses
            if close[i] > weekly_r1_aligned[i] or close[i] > ema50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals