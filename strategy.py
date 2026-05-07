#!/usr/bin/env python3
# 1D_Camarilla_R3_S3_Breakout_WeeklyTrend_Volume
# Hypothesis: Uses weekly CAMARILLA R3/S3 levels from weekly timeframe with daily price action for entry and weekly EMA trend filter.
# Enters long when daily close breaks above weekly R3 in uptrend (daily close > weekly EMA34) with volume confirmation.
# Enters short when daily close breaks below weekly S3 in downtrend (daily close < weekly EMA34) with volume confirmation.
# Exits when price returns inside the weekly pivot range (S3 to R3).
# Weekly timeframe reduces noise, daily entries improve timing, volume confirmation filters false breakouts.
# Designed for low trade frequency (<25/year) and works in both bull and bear markets by following weekly trend.

name = "1D_Camarilla_R3_S3_Breakout_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0

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
    
    # Get weekly data for CAMARILLA calculation and trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    # Calculate weekly OHLC for CAMARILLA pivots
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    # CAMARILLA levels: R3 = close + (high - low) * 1.1/4, S3 = close - (high - low) * 1.1/4
    rng = high_weekly - low_weekly
    camarilla_r3 = close_weekly + rng * 1.1 / 4
    camarilla_s3 = close_weekly - rng * 1.1 / 4
    
    # Weekly EMA34 for trend filter
    ema34_weekly = pd.Series(close_weekly).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align CAMARILLA levels and EMA to daily timeframe
    r3_aligned = align_htf_to_ltf(prices, df_weekly, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_weekly, camarilla_s3)
    ema34_aligned = align_htf_to_ltf(prices, df_weekly, ema34_weekly)
    
    # Volume filter: current volume > 1.5x average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure we have volume MA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema34_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: spike confirmation
        volume_filter = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: Price breaks above weekly R3 + Uptrend (daily close > weekly EMA34) + volume confirmation
            if (close[i] > r3_aligned[i] and 
                close[i] > ema34_aligned[i] and
                volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below weekly S3 + Downtrend (daily close < weekly EMA34) + volume confirmation
            elif (close[i] < s3_aligned[i] and 
                  close[i] < ema34_aligned[i] and
                  volume_filter):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price returns inside weekly pivot range (below R3 and above S3)
            if close[i] < r3_aligned[i] and close[i] > s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price returns inside weekly pivot range (below R3 and above S3)
            if close[i] < r3_aligned[i] and close[i] > s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals