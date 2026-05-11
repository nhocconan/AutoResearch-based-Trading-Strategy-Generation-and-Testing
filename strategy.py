#!/usr/bin/env python3
# 12h_Camarilla_Pivot_R3S3_Breakout_1dTrend_Volume_Spike
# Hypothesis: 12h timeframe with 1d Camarilla R3/S3 breakout, 1d EMA34 trend filter, and volume spike confirmation.
# Targets 15-30 trades/year to minimize fee drag. Uses breakout logic with trend alignment to capture momentum in both bull and bear markets.
# Volume spike reduces false breakouts. Designed for low turnover and high conviction trades.

name = "12h_Camarilla_Pivot_R3S3_Breakout_1dTrend_Volume_Spike"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 1d Camarilla Pivot Levels (R3/S3) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    r3 = pivot + (range_1d * 1.1 / 2)
    s3 = pivot - (range_1d * 1.1 / 2)
    
    # Align 1d levels to 12h
    r3_12h = align_htf_to_ltf(prices, df_1d, r3)
    s3_12h = align_htf_to_ltf(prices, df_1d, s3)
    
    # === 1d EMA34 Trend Filter ===
    ema34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema34_1d_12h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # === Volume Spike Filter (20-period EMA) ===
    vol_ema20 = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ok = volume > vol_ema20 * 2.0  # Require 2x average volume for confirmation
    
    # === Signal Parameters ===
    position_size = 0.25  # 25% position size to manage risk
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (covers EMA34)
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(r3_12h[i]) or np.isnan(s3_12h[i]) or 
            np.isnan(ema34_1d_12h[i]) or np.isnan(volume_ok[i])):
            if position == 1:
                signals[i] = 0.0
            elif position == -1:
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Break above R3 + above 1d EMA34 + volume spike
            if (close[i] > r3_12h[i] and 
                close[i] > ema34_1d_12h[i] and 
                volume_ok[i]):
                signals[i] = position_size
                position = 1
            # Short: Break below S3 + below 1d EMA34 + volume spike
            elif (close[i] < s3_12h[i] and 
                  close[i] < ema34_1d_12h[i] and 
                  volume_ok[i]):
                signals[i] = -position_size
                position = -1
        else:
            # Hold position until opposite level is broken
            if position == 1:
                # Exit: Price closes below S3 (opposite level)
                if close[i] < s3_12h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                # Exit: Price closes above R3 (opposite level)
                if close[i] > r3_12h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size
    
    return signals