#!/usr/bin/env python3
# 12h_Camarilla_R3S3_Breakout_1wTrend_Volume_Spike
# Hypothesis: 1w Camarilla R3/S3 levels provide strong weekly support/resistance.
# Breakouts above R3 (long) or below S3 (short) are traded only when aligned with 1w EMA200 trend
# and confirmed by volume spikes, with exits on opposite level touches.
# Designed for low turnover (~15-25 trades/year) to minimize fee drag in 2025 bear/ranging markets.
# Target: 50-150 total trades over 4 years to avoid excessive fee drag.

name = "12h_Camarilla_R3S3_Breakout_1wTrend_Volume_Spike"
timeframe = "12h"
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
    
    # === 1w Camarilla Pivot Levels ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    r3_1w = pivot_1w + (range_1w * 1.1 / 2)
    s3_1w = pivot_1w - (range_1w * 1.1 / 2)
    
    # Align 1w Camarilla levels to 12h
    r3_12h = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_12h = align_htf_to_ltf(prices, df_1w, s3_1w)
    
    # === 1w EMA200 Trend Filter ===
    close_1w = df_1w['close'].values
    ema200_1w = pd.Series(close_1w).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema200_12h = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # === Volume Spike Filter (20-period EMA) ===
    vol_ema20 = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ok = volume > vol_ema20 * 2.0  # Require 2x average volume
    
    # === Signal Parameters ===
    position_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for indicators)
    start_idx = 200  # covers EMA200
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(r3_12h[i]) or np.isnan(s3_12h[i]) or 
            np.isnan(ema200_12h[i]) or np.isnan(volume_ok[i])):
            if position == 1:
                signals[i] = 0.0
            elif position == -1:
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Break above R3 + above 1w EMA200 + volume spike
            if close[i] > r3_12h[i] and close[i] > ema200_12h[i] and volume_ok[i]:
                signals[i] = position_size
                position = 1
            # Short: Break below S3 + below 1w EMA200 + volume spike
            elif close[i] < s3_12h[i] and close[i] < ema200_12h[i] and volume_ok[i]:
                signals[i] = -position_size
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit: Price touches or crosses below S3 (opposite level)
                if close[i] < s3_12h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                # Exit: Price touches or crosses above R3 (opposite level)
                if close[i] > r3_12h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size
    
    return signals