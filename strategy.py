#!/usr/bin/env python3
"""
12h_12h_1w_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike_v1
Hypothesis: Use Camarilla R3/S3 levels from weekly timeframe as significant support/resistance.
Price breaking above R3 with volume spike indicates bullish momentum; breaking below S3 with volume spike indicates bearish momentum.
Weekly EMA50 trend filter ensures alignment with higher timeframe trend.
Target: 50-150 total trades over 4 years on 12h timeframe.
"""

name = "12h_12h_1w_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike_v1"
timeframe = "12h"
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
    
    # === WEEKLY Data for Camarilla and Trend ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla levels for weekly timeframe
    # Based on previous week's OHLC
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Camarilla levels: R3, S3
    # R3 = close + (high - low) * 1.1/4
    # S3 = close - (high - low) * 1.1/4
    range_1w = high_1w - low_1w
    r3_1w = close_1w + range_1w * 1.1 / 4
    s3_1w = close_1w - range_1w * 1.1 / 4
    
    # Weekly EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly data to 12h timeframe
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume spike detection: current volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(r3_1w_aligned[i]) or 
            np.isnan(s3_1w_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price crosses above R3 with volume spike AND price above weekly EMA50 (uptrend)
            if close[i] > r3_1w_aligned[i] and volume_spike[i] and close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below S3 with volume spike AND price below weekly EMA50 (downtrend)
            elif close[i] < s3_1w_aligned[i] and volume_spike[i] and close[i] < ema50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below S3 OR price crosses below weekly EMA50
            if close[i] < s3_1w_aligned[i] or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price crosses above R3 OR price crosses above weekly EMA50
            if close[i] > r3_1w_aligned[i] or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals