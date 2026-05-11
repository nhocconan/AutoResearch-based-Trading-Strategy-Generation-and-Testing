#!/usr/bin/env python3
"""
4h_Camarilla_Pivot_R3S3_Breakout_12hTrend_Volume_Spike
Hypothesis: Camarilla R3/S3 levels from 1d provide strong intraday support/resistance.
Breakouts above R3 (long) or below S3 (short) are traded only when aligned with 12h EMA50 trend
and confirmed by volume spikes, with exits on opposite Camarilla level touches.
Designed for low turnover (~20-30 trades/year) to minimize fee drag in ranging 2025 markets.
"""

name = "4h_Camarilla_Pivot_R3S3_Breakout_12hTrend_Volume_Spike"
timeframe = "4h"
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
    
    # === 1d Camarilla Pivot Levels ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    r3 = pivot + (range_1d * 1.1 / 2)
    s3 = pivot - (range_1d * 1.1 / 2)
    
    # Align 1d Camarilla levels to 4h
    r3_4h = align_htf_to_ltf(prices, df_1d, r3)
    s3_4h = align_htf_to_ltf(prices, df_1d, s3)
    
    # === 12h EMA50 Trend Filter ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_12h_4h = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # === Volume Spike Filter (20-period EMA) ===
    vol_ema20 = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ok = volume > vol_ema20 * 2.0  # Require 2x average volume
    
    # === Signal Parameters ===
    position_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for indicators)
    start_idx = 50  # covers EMA50
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(r3_4h[i]) or np.isnan(s3_4h[i]) or 
            np.isnan(ema50_12h_4h[i]) or np.isnan(volume_ok[i])):
            if position == 1:
                signals[i] = 0.0
            elif position == -1:
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Break above R3 + above 12h EMA50 + volume spike
            if close[i] > r3_4h[i] and close[i] > ema50_12h_4h[i] and volume_ok[i]:
                signals[i] = position_size
                position = 1
            # Short: Break below S3 + below 12h EMA50 + volume spike
            elif close[i] < s3_4h[i] and close[i] < ema50_12h_4h[i] and volume_ok[i]:
                signals[i] = -position_size
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit: Price touches or crosses below S3 (opposite level)
                if close[i] < s3_4h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                # Exit: Price touches or crosses above R3 (opposite level)
                if close[i] > r3_4h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size
    
    return signals