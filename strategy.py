#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Pivot_Breakout_1wTrend
Hypothesis: Use daily Camarilla R3/S3 levels as key reversal levels with weekly trend filter (price above/below 50-week EMA). 
Go long when price breaks above R3 with volume confirmation in bullish weekly trend, short when price breaks below S3 in bearish weekly trend.
Camarilla levels work well in ranging markets (common in 2025-2026) while the weekly trend filter prevents counter-trend trades.
Designed for 12h timeframe to target 12-37 trades/year with low frequency and high win rate.
"""

name = "12h_Camarilla_R3_S3_Pivot_Breakout_1wTrend"
timeframe = "12h"
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
    
    # Get daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla formula: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low)
    # S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    cam_r3 = close_1d + 1.1 * (high_1d - low_1d)
    cam_s3 = close_1d - 1.1 * (high_1d - low_1d)
    
    # Align Camarilla levels to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, cam_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, cam_s3)
    
    # Get weekly EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate volume average (24-period = 12 days) for volume spike filter
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike condition: current volume > 1.3x 24-period average
        vol_spike = volume[i] > 1.3 * vol_ma_24[i]
        
        if position == 0:
            # LONG: Price breaks above R3 with volume spike in bullish weekly trend
            if close[i] > r3_aligned[i] and vol_spike and close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 with volume spike in bearish weekly trend
            elif close[i] < s3_aligned[i] and vol_spike and close[i] < ema_50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S3 or weekly trend turns bearish
            if close[i] < s3_aligned[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R3 or weekly trend turns bullish
            if close[i] > r3_aligned[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals