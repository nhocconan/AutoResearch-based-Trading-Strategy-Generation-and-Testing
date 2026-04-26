#!/usr/bin/env python3
"""
6h_WeeklyPivot_VolumeReversion_v1
Hypothesis: Trade weekly Camarilla pivot mean reversion on 6h timeframe with volume confirmation. 
Long when price touches S3 with volume spike (>2x median) and closes above open (bullish rejection).
Short when price touches R3 with volume spike and closes below open (bearish rejection).
Uses 1d EMA50 as trend filter: only long when price > EMA50, short when price < EMA50.
Weekly pivot provides structural support/resistance that works in both bull and bear markets.
Volume spike confirms institutional interest at pivot levels.
Target: 50-150 total trades over 4 years (12-37/year) with discrete position sizing (0.25).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Camarilla pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Get daily data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Weekly Camarilla levels from previous weekly bar
    # R3 = close + 1.1*(high-low)*1.1/4
    # S3 = close - 1.1*(high-low)*1.1/4
    prev_weekly_close = df_1w['close'].shift(1).values
    prev_weekly_high = df_1w['high'].shift(1).values
    prev_weekly_low = df_1w['low'].shift(1).values
    camarilla_r3 = prev_weekly_close + 1.1 * (prev_weekly_high - prev_weekly_low) * 1.1 / 4
    camarilla_s3 = prev_weekly_close - 1.1 * (prev_weekly_high - prev_weekly_low) * 1.1 / 4
    
    # Daily EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF data to 6h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: 2.0x median volume (balanced for trade frequency)
    vol_median = pd.Series(volume).rolling(window=30, min_periods=30).median().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of weekly pivot calculation (2), daily EMA (50), volume median (30)
    start_idx = max(2, 50, 30)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_median[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        camarilla_r3_val = camarilla_r3_aligned[i]
        camarilla_s3_val = camarilla_s3_aligned[i]
        ema_50_val = ema_50_1d_aligned[i]
        close_val = close[i]
        open_val = open_price[i]
        volume_val = volume[i]
        vol_median_val = vol_median[i]
        
        if position == 0:
            # Long: price touches S3 with volume spike and bullish rejection (close > open)
            # Only in uptrend (price > EMA50)
            long_signal = (low_val <= camarilla_s3_val) and \
                          (close_val > open_val) and \
                          (volume_val > 2.0 * vol_median_val) and \
                          (close_val > ema_50_val)
            
            # Short: price touches R3 with volume spike and bearish rejection (close < open)
            # Only in downtrend (price < EMA50)
            short_signal = (high_val >= camarilla_r3_val) and \
                           (close_val < open_val) and \
                           (volume_val > 2.0 * vol_median_val) and \
                           (close_val < ema_50_val)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit conditions: price reaches midpoint (mean reversion target) or trend reversal
            midpoint = (camarilla_s3_val + camarilla_r3_val) / 2
            if close_val >= midpoint or close_val < ema_50_val:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit conditions: price reaches midpoint (mean reversion target) or trend reversal
            midpoint = (camarilla_s3_val + camarilla_r3_val) / 2
            if close_val <= midpoint or close_val > ema_50_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WeeklyPivot_VolumeReversion_v1"
timeframe = "6h"
leverage = 1.0