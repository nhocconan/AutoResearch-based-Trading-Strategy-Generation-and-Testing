#!/usr/bin/env python3
"""
6h_Camarilla_R3_S3_Breakout_12hTrend_VolumeSpike
Hypothesis: Fade at Camarilla R3/S3 levels with 12h trend filter and volume confirmation. Works in bull/bear by fading overextended moves within the trend context. Uses 12h EMA50 for trend direction and volume spikes for confirmation. Targets 15-30 trades/year to minimize fee drag.
"""

name = "6h_Camarilla_R3_S3_Breakout_12hTrend_VolumeSpike"
timeframe = "6h"
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
    
    # === 12h EMA50 Trend Filter ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_12h_6h = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # === Volume Spike Filter ===
    vol_ema20 = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ok = volume > vol_ema20 * 1.5  # 1.5x average volume
    
    # === Camarilla Levels from 1d (using previous day's range) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels: R3, S3
    # R3 = close + 1.1 * (high - low) / 6
    # S3 = close - 1.1 * (high - low) / 6
    camarilla_range = high_1d - low_1d
    r3 = close_1d + 1.1 * camarilla_range / 6
    s3 = close_1d - 1.1 * camarilla_range / 6
    
    # Align to 6h timeframe
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    
    # === Signal Parameters ===
    position_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (covers EMA and Camarilla calculation)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(ema50_12h_6h[i]) or np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or 
            np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price touches/bounces from S3 support + above 12h EMA50 + volume spike
            if (close[i] <= s3_6h[i] * 1.001 and  # Allow small tolerance for touch
                close[i] > ema50_12h_6h[i] and 
                volume_ok[i]):
                signals[i] = position_size
                position = 1
            # Short: Price touches/rejects from R3 resistance + below 12h EMA50 + volume spike
            elif (close[i] >= r3_6h[i] * 0.999 and  # Allow small tolerance for touch
                  close[i] < ema50_12h_6h[i] and 
                  volume_ok[i]):
                signals[i] = -position_size
                position = -1
        else:
            # Exit: Price moves back through the opposite Camarilla level
            if position == 1:
                if close[i] >= r3_6h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                if close[i] <= s3_6h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size
    
    return signals