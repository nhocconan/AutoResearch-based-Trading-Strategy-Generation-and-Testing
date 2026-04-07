#!/usr/bin/env python3
"""
12h_camarilla_pivot_1w_trend_volume_v1
Hypothesis: Use daily Camarilla pivot levels as support/resistance on 12h timeframe, filtered by weekly trend (price above/below weekly EMA50) and confirmed by volume spikes. Enter long at S3 support in uptrend, short at R3 resistance in downtrend. Exit at opposite pivot level. Weekly trend filter prevents counter-trend trades. Target: 50-150 trades over 4 years (12-37/year) to minimize fee drag while capturing meaningful swings.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1w_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate daily Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Use previous day's OHLC for today's Camarilla levels
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate pivot and levels
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    S1 = prev_close - (range_hl * 1.1 / 12)
    S2 = prev_close - (range_hl * 1.1 / 6)
    S3 = prev_close - (range_hl * 1.1 / 4)
    R1 = prev_close + (range_hl * 1.1 / 12)
    R2 = prev_close + (range_hl * 1.1 / 6)
    R3 = prev_close + (range_hl * 1.1 / 4)
    
    # Align daily levels to 12h
    pivot_12h = align_htf_to_ltf(prices, df_1d, pivot)
    S1_12h = align_htf_to_ltf(prices, df_1d, S1)
    S2_12h = align_htf_to_ltf(prices, df_1d, S2)
    S3_12h = align_htf_to_ltf(prices, df_1d, S3)
    R1_12h = align_htf_to_ltf(prices, df_1d, R1)
    R2_12h = align_htf_to_ltf(prices, df_1d, R2)
    R3_12h = align_htf_to_ltf(prices, df_1d, R3)
    
    # Weekly trend filter: EMA50
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Volume spike detection: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any data is NaN
        if (np.isnan(pivot_12h[i]) or np.isnan(S3_12h[i]) or np.isnan(R3_12h[i]) or 
            np.isnan(ema_50_12h[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        # Weekly trend: price above/below EMA50
        weekly_uptrend = close[i] > ema_50_12h[i]
        
        if position == 1:  # Long position
            # Exit: price reaches R3 (resistance) or closes below S3 (stop)
            if close[i] >= R3_12h[i] or close[i] <= S3_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches S3 (support) or closes above R3 (stop)
            if close[i] <= S3_12h[i] or close[i] >= R3_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only enter on volume spike and with weekly trend alignment
            if volume_spike[i]:
                if weekly_uptrend:
                    # Long: price at or below S3 support
                    if close[i] <= S3_12h[i]:
                        position = 1
                        signals[i] = 0.25
                else:
                    # Short: price at or above R3 resistance
                    if close[i] >= R3_12h[i]:
                        position = -1
                        signals[i] = -0.25
    
    return signals