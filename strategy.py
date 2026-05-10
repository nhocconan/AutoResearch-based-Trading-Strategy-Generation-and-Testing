#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1dTrend_Volume
Hypothesis: Use 12h Camarilla R3/S3 levels as dynamic breakout levels filtered by 1d EMA trend and volume confirmation.
Camarilla levels identify key support/resistance; breakouts above R3 or below S3 with volume capture strong momentum.
Trend filter (1d EMA100) avoids counter-trend trades. Designed for 20-30 trades/year, works in bull/bear via trend filter.
"""

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data for EMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    # Get 12h data for Camarilla calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h Camarilla levels (using prior day)
    high_prev = df_12h['high'].shift(1).values
    low_prev = df_12h['low'].shift(1).values
    close_prev = df_12h['close'].shift(1).values
    
    # Camarilla formula: R3 = close + 1.1*(high-low), S3 = close - 1.1*(high-low)
    camarilla_r3 = close_prev + 1.1 * (high_prev - low_prev)
    camarilla_s3 = close_prev - 1.1 * (high_prev - low_prev)
    
    # Align Camarilla levels to 12h timeframe (wait for 12h bar to close)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    
    # 1d EMA100 for trend filter
    ema_100 = pd.Series(df_1d['close'].values).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema_100_aligned = align_htf_to_ltf(prices, df_1d, ema_100)
    
    # Get 12h price and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Camarilla (needs 12h bar), EMA100 (100 bars), volume EMA (20)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema_100_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: above EMA100 (uptrend) AND price breaks above Camarilla R3 with volume
            if close[i] > ema_100_aligned[i] and high[i] > camarilla_r3_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: below EMA100 (downtrend) AND price breaks below Camarilla S3 with volume
            elif close[i] < ema_100_aligned[i] and low[i] < camarilla_s3_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below Camarilla S3 OR trend turns bearish
            if low[i] < camarilla_s3_aligned[i] or close[i] < ema_100_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Camarilla R3 OR trend turns bullish
            if high[i] > camarilla_r3_aligned[i] or close[i] > ema_100_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals