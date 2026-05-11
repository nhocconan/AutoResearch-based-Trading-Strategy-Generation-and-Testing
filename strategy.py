#!/usr/bin/env python3
"""
4h_12h_Camarilla_Pivot_Volume_Trend
Hypothesis: Camarilla pivot levels on 12h combined with volume confirmation and trend filter.
Long when: price breaks above R3 with volume > 20-period average and 12h trend up.
Short when: price breaks below S3 with volume > 20-period average and 12h trend down.
Exit when: price returns to mean (central pivot) or trend reverses.
Camarilla levels work in both bull and bear markets as they adapt to volatility.
Using 12h pivots on 4h chart reduces noise and improves signal quality.
Targets 20-40 trades/year (80-160 over 4 years) to minimize fee drag.
"""

name = "4h_12h_Camarilla_Pivot_Volume_Trend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 12h data for Camarilla pivots
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # 4h OHLCV
    close_4h = prices['close'].values
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    volume_4h = prices['volume'].values
    
    # --- 12h Camarilla Pivot Levels ---
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate pivots for each 12h bar
    pivot = (high_12h + low_12h + close_12h) / 3.0
    range_12h = high_12h - low_12h
    
    # Camarilla levels
    R4 = close_12h + range_12h * 1.1 / 2
    R3 = close_12h + range_12h * 1.1 / 4
    R2 = close_12h + range_12h * 1.1 / 6
    R1 = close_12h + range_12h * 1.1 / 12
    S1 = close_12h - range_12h * 1.1 / 12
    S2 = close_12h - range_12h * 1.1 / 6
    S3 = close_12h - range_12h * 1.1 / 4
    S4 = close_12h - range_12h * 1.1 / 2
    
    # Align pivots to 4h timeframe
    R3_12h = align_htf_to_ltf(prices, df_12h, R3)
    S3_12h = align_htf_to_ltf(prices, df_12h, S3)
    pivot_12h = align_htf_to_ltf(prices, df_12h, pivot)
    
    # --- 12h Trend Filter: EMA50 ---
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # --- Volume Confirmation: 4h volume > 20-period average ---
    vol_ma_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50  # for EMA50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(R3_12h[i]) or np.isnan(S3_12h[i]) or 
            np.isnan(pivot_12h[i]) or np.isnan(ema50_12h_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 12h trend
        trend_up = close_4h[i] > ema50_12h_aligned[i]
        trend_down = close_4h[i] < ema50_12h_aligned[i]
        
        # Volume confirmation
        vol_ok = volume_4h[i] > vol_ma_20[i]
        
        if position == 0:
            # Look for entries only in direction of 12h trend with volume
            if close_4h[i] > R3_12h[i] and trend_up and vol_ok:
                # Long: price above R3 + 12h uptrend + volume
                signals[i] = 0.25
                position = 1
            elif close_4h[i] < S3_12h[i] and trend_down and vol_ok:
                # Short: price below S3 + 12h downtrend + volume
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: price returns to pivot OR trend turns down
                if close_4h[i] <= pivot_12h[i] or not trend_up:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price returns to pivot OR trend turns up
                if close_4h[i] >= pivot_12h[i] or not trend_down:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals