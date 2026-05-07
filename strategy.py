#!/usr/bin/env python3
"""
12h_Camarilla_R3S3_Breakout_1wTrend_Volume
Hypothesis: Use weekly Camarilla pivot levels (R3/S3) from daily data as key support/resistance. Enter long when price breaks above R3 with volume confirmation in weekly uptrend, short when price breaks below S3 with volume confirmation in weekly downtrend. Exit on opposite level break. Designed for 12h to capture multi-week swings with low frequency, suitable for both bull and bear markets via trend filter.
"""

name = "12h_Camarilla_R3S3_Breakout_1wTrend_Volume"
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
    
    # Get daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    # Classic Camarilla: 
    # H4 = C + 1.5*(H-L), L4 = C - 1.5*(H-L)
    # R3 = C + 1.25*(H-L), S3 = C - 1.25*(H-L)
    # We use previous day's H,L,C
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Avoid first day
    valid_prev = ~(np.isnan(prev_high) | np.isnan(prev_low) | np.isnan(prev_close))
    
    # Calculate levels
    hl_range = prev_high - prev_low
    R3 = prev_close + 1.25 * hl_range
    S3 = prev_close - 1.25 * hl_range
    
    # Align to 12h timeframe (no extra delay needed as Camarilla uses previous day's data)
    R3_12h = align_htf_to_ltf(prices, df_1d, R3)
    S3_12h = align_htf_to_ltf(prices, df_1d, S3)
    
    # Get weekly data for trend filter (EMA50 on weekly close)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Weekly close aligned for trend comparison
    weekly_close_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
    
    # Volume confirmation: 50-period average on 12h
    vol_ma50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    vol_ratio = np.divide(volume, vol_ma50, out=np.zeros_like(volume), where=vol_ma50!=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for weekly EMA50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(R3_12h[i]) or np.isnan(S3_12h[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(weekly_close_aligned[i]) or
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly trend determination
        weekly_trend_up = weekly_close_aligned[i] > ema_50_1w_aligned[i]
        weekly_trend_down = weekly_close_aligned[i] < ema_50_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above R3 with volume confirmation in weekly uptrend
            if (close[i] > R3_12h[i] and
                vol_ratio[i] > 2.0 and 
                weekly_trend_up):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 with volume confirmation in weekly downtrend
            elif (close[i] < S3_12h[i] and 
                  vol_ratio[i] > 2.0 and 
                  weekly_trend_down):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below S3 (support level)
            if close[i] < S3_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above R3 (resistance level)
            if close[i] > R3_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals