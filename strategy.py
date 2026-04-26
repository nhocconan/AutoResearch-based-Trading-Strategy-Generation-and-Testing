#!/usr/bin/env python3
"""
6h_Camarilla_R3_S3_Fade_1wTrend_VolumeFilter
Hypothesis: Fade at weekly Camarilla R3/S3 levels with 1w trend filter and volume confirmation on 6h timeframe.
Long when price touches S3 in 1w uptrend with volume spike. Short when price touches R3 in 1w downtrend with volume spike.
Uses discrete position sizing (0.25) to minimize fee churn. Designed for mean reversion in ranging markets and 
breakout continuation in strong trends via weekly structure.
Target: 12-37 trades/year (50-150 total over 4 years).
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
    
    # Get 1d data for weekly Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate weekly Camarilla pivots from prior week's OHLC
    # We'll use rolling window of 5 days (1 week) to get weekly OHLC
    # Since we have daily data, we can approximate weekly by taking last 5 days
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Rolling window for weekly high, low, close (prior complete week)
    # Use shift(1) to avoid look-ahead - use prior week's data
    weekly_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().shift(1).values
    weekly_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().shift(1).values
    weekly_close = pd.Series(close_1d).rolling(window=5, min_periods=5).last().shift(1).values
    
    # Camarilla calculations for R3 and S3
    # R3 = close + (high - low) * 1.1/4
    # S3 = close - (high - low) * 1.1/4
    camarilla_range = weekly_high - weekly_low
    r3 = weekly_close + camarilla_range * 1.1 / 4
    s3 = weekly_close - camarilla_range * 1.1 / 4
    
    # Align weekly Camarilla levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # 1w trend filter: use weekly EMA20 on close
    weekly_ema20 = pd.Series(weekly_close).ewm(span=20, min_periods=20, adjust=False).mean().values
    weekly_ema20_aligned = align_htf_to_ltf(prices, df_1d, weekly_ema20)
    uptrend_1w = weekly_close > weekly_ema20  # Weekly trend based on prior week's data
    downtrend_1w = weekly_close < weekly_ema20
    uptrend_1w_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1w.astype(float))
    downtrend_1w_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1w.astype(float))
    
    # Volume confirmation: volume > 1.8x 20-period MA (slightly higher threshold for fewer trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 5 for weekly calcs + 20 for EMA + 20 for volume MA)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(weekly_ema20_aligned[i]) or np.isnan(uptrend_1w_aligned[i]) or 
            np.isnan(downtrend_1w_aligned[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:
            # Long: price touches or goes below S3 in 1w uptrend with volume spike
            if (low[i] <= s3_aligned[i] and 
                uptrend_1w_aligned[i] > 0.5 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price touches or goes above R3 in 1w downtrend with volume spike
            elif (high[i] >= r3_aligned[i] and 
                  downtrend_1w_aligned[i] > 0.5 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price reaches midpoint (neutral level) or trend changes
            midpoint = (r3_aligned[i] + s3_aligned[i]) / 2
            if (close[i] >= midpoint or 
                uptrend_1w_aligned[i] <= 0.5):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price reaches midpoint or trend changes
            midpoint = (r3_aligned[i] + s3_aligned[i]) / 2
            if (close[i] <= midpoint or 
                downtrend_1w_aligned[i] <= 0.5):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Camarilla_R3_S3_Fade_1wTrend_VolumeFilter"
timeframe = "6h"
leverage = 1.0