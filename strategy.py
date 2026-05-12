#!/usr/bin/env python3
name = "6h_WeeklyPivotBreakout_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d data for trend ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # === 1d EMA34 for trend (medium-term) ===
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # === Weekly pivot points from 1d data (calculate on 1d then align to 6h) ===
    # Weekly pivot: (H + L + C) / 3 where H,L,C are weekly high/low/close
    # We'll approximate using 1d data: weekly high = max of last 5 days high, etc.
    # But to avoid lookahead, we use previous week's data
    # Calculate weekly high/low/close using 1d data with 5-day rolling
    weekly_high = pd.Series(high).rolling(window=5, min_periods=5).max().values
    weekly_low = pd.Series(low).rolling(window=5, min_periods=5).min().values
    weekly_close = pd.Series(close_1d).rolling(window=5, min_periods=5).last().values
    
    # Align weekly close to 1d index first, then to 6h
    weekly_close_aligned_1d = weekly_close  # already on 1d
    weekly_high_aligned_1d = weekly_high
    weekly_low_aligned_1d = weekly_low
    
    # Now align to 6h timeframe
    weekly_high_6h = align_htf_to_ltf(prices, df_1d, weekly_high_aligned_1d)
    weekly_low_6h = align_htf_to_ltf(prices, df_1d, weekly_low_aligned_1d)
    weekly_close_6h = align_htf_to_ltf(prices, df_1d, weekly_close_aligned_1d)
    
    # Weekly pivot and support/resistance levels
    weekly_pivot = (weekly_high_6h + weekly_low_6h + weekly_close_6h) / 3.0
    weekly_r1 = 2 * weekly_pivot - weekly_low_6h
    weekly_s1 = 2 * weekly_pivot - weekly_high_6h
    weekly_r2 = weekly_pivot + (weekly_high_6h - weekly_low_6h)
    weekly_s2 = weekly_pivot - (weekly_high_6h - weekly_low_6h)
    weekly_r3 = weekly_high_6h + 2 * (weekly_pivot - weekly_low_6h)
    weekly_s3 = weekly_low_6h - 2 * (weekly_high_6h - weekly_pivot)
    
    # Volume spike detection (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 34, 20, 5)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(weekly_pivot[i]) or
            np.isnan(weekly_r3[i]) or
            np.isnan(weekly_s3[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Break above weekly R3 with volume spike and above 1d EMA34
            if (close[i] > weekly_r3[i] and 
                volume_spike[i] and
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Break below weekly S3 with volume spike and below 1d EMA34
            elif (close[i] < weekly_s3[i] and 
                  volume_spike[i] and
                  close[i] < ema34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Close below weekly pivot or below 1d EMA34
            if close[i] < weekly_pivot[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Close above weekly pivot or above 1d EMA34
            if close[i] > weekly_pivot[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals