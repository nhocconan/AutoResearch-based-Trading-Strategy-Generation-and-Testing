# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Hypothesis: 4h timeframe with 12h trend filter. Use Camarilla pivot levels (R3/S3) from 12h as breakout levels,
confirmed by volume spike and 12h EMA50 trend direction. Avoids overtrading by requiring confluence of 3 conditions.
Designed to work in both bull and bear markets via trend filter and volatility-based breakout.
"""

name = "4h_Camarilla_R3S3_Breakout_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0

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
    
    # Get 12h data for Camarilla pivots and trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla levels on 12h close (using prior 12h bar's OHLC)
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), etc.
    # We use R3 and S3 as breakout levels
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate pivot range (high-low) for each 12h bar
    range_12h = high_12h - low_12h
    
    # Camarilla levels based on prior bar's OHLC
    r3_12h = close_12h + 1.1 * range_12h
    s3_12h = close_12h - 1.1 * range_12h
    
    # Align Camarilla levels to 4h timeframe (no extra delay - levels are known at bar close)
    r3_12h_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    s3_12h_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: current 4h volume vs 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isclose(r3_12h_aligned[i], 0) or np.isclose(s3_12h_aligned[i], 0) or
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend direction from 12h EMA50
        # Uptrend: price above EMA50, Downtrend: price below EMA50
        # We'll use the 12h close aligned to 4h for trend determination
        close_12h_aligned = align_htf_to_ltf(prices, df_12h, close_12h)
        if np.isnan(close_12h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        trend_up = close_12h_aligned[i] > ema_50_12h_aligned[i]
        trend_down = close_12h_aligned[i] < ema_50_12h_aligned[i]
        
        if position == 0:
            # Long breakout: price breaks above R3 with volume confirmation and uptrend
            if (close[i] > r3_12h_aligned[i] and 
                volume_ratio[i] > 1.5 and 
                trend_up):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below S3 with volume confirmation and downtrend
            elif (close[i] < s3_12h_aligned[i] and 
                  volume_ratio[i] > 1.5 and 
                  trend_down):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below S3 or trend changes to down
            if (close[i] < s3_12h_aligned[i] or 
                not trend_up):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above R3 or trend changes to up
            if (close[i] > r3_12h_aligned[i] or 
                not trend_down):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals