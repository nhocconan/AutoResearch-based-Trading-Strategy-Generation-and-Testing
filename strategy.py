#!/usr/bin/env python3
# 6h_1w1d_Camarilla_R3_S3_Breakout_WeeklyTrend_Volume
# Hypothesis: 6h Camarilla R3/S3 breakout with weekly trend filter (price > weekly EMA50) and volume confirmation.
# Enters long when price breaks above R3 in bullish weekly trend with volume surge, short when breaks below S3 in bearish weekly trend.
# Uses weekly timeframe for trend filter to avoid whipsaws in both bull and bear markets.
# Targets low trade frequency (12-37/year) to minimize fee drag.

name = "6h_1w1d_Camarilla_R3_S3_Breakout_WeeklyTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 6h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Calculate daily data for Camarilla levels (R3, S3)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla for each day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R3 = close + 1.1*(high-low)*2
    # Camarilla S3 = close - 1.1*(high-low)*2
    r3 = close_1d + 1.1 * (high_1d - low_1d) * 2
    s3 = close_1d - 1.1 * (high_1d - low_1d) * 2
    
    # Align R3 and S3 to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume confirmation (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_50_aligned[i]) or
            np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly trend filter
        bullish_trend = close_1w[i] > ema_50[i]  # Use raw weekly close for trend
        bearish_trend = close_1w[i] < ema_50[i]
        
        # Volume confirmation (2.0x average)
        volume_surge = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: breakout above R3 in bullish weekly trend with volume
            if close[i] > r3_aligned[i] and bullish_trend and volume_surge:
                signals[i] = 0.25
                position = 1
            # Short: breakdown below S3 in bearish weekly trend with volume
            elif close[i] < s3_aligned[i] and bearish_trend and volume_surge:
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Long exit: price closes below S1 (reversion to mean)
                # Calculate S1 for exit
                s1 = close_1d - 1.1 * (high_1d - low_1d) / 12
                s1_aligned_exit = align_htf_to_ltf(prices, df_1d, s1)
                if not np.isnan(s1_aligned_exit[i]) and close[i] < s1_aligned_exit[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Short exit: price closes above R1 (reversion to mean)
                # Calculate R1 for exit
                r1 = close_1d + 1.1 * (high_1d - low_1d) / 12
                r1_aligned_exit = align_htf_to_ltf(prices, df_1d, r1)
                if not np.isnan(r1_aligned_exit[i]) and close[i] > r1_aligned_exit[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals