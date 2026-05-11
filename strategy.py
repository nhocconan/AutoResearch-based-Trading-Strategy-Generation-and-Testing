#!/usr/bin/env python3
"""
12h_1d_1w_Camarilla_Pivot_Breakout_TrendVol
Hypothesis: Use 12h timeframe for trading with 1d and 1w filters.
- Trend: Price above/below weekly (1w) 20-period EMA (macro trend)
- Entry: 12h price breaks above/below daily (1d) Camarilla R3/S3 levels
- Filter: Require volume spike (1.5x 20-period average volume) on breakout
- Exit: Reverse signal or trailing stop via EMA(50) on 12h
- Position size: 0.25 (discrete to reduce churn)
- Designed for fewer trades (~20-40/year) to avoid fee drag, works in bull/bear via trend filter.
"""

name = "12h_1d_1w_Camarilla_Pivot_Breakout_TrendVol"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla levels: R3, R2, R1, PP, S1, S2, S3"""
    range_val = high - low
    pp = (high + low + close) / 3.0
    r3 = pp + (high - low) * 1.1 / 4
    s3 = pp - (high - low) * 1.1 / 4
    return r3, s3

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Get weekly data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 12h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Daily Camarilla R3/S3 levels ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_r3 = np.full(len(high_1d), np.nan)
    camarilla_s3 = np.full(len(low_1d), np.nan)
    
    for i in range(len(high_1d)):
        r3, s3 = calculate_camarilla(high_1d[i], low_1d[i], close_1d[i])
        camarilla_r3[i] = r3
        camarilla_s3[i] = s3
    
    # Align daily Camarilla levels to 12h timeframe
    camarilla_r3_12h = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_12h = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # --- Weekly EMA50 for trend filter ---
    close_1w = df_1w['close'].values
    ema_50_1w = np.full(len(close_1w), np.nan)
    
    if len(close_1w) >= 50:
        ema_50_1w[49] = np.mean(close_1w[:50])  # Simple average for first value
        multiplier = 2 / (50 + 1)
        for i in range(50, len(close_1w)):
            ema_50_1w[i] = (close_1w[i] * multiplier) + (ema_50_1w[i-1] * (1 - multiplier))
    
    # Align weekly EMA50 to 12h timeframe
    ema_50_1w_12h = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # --- Volume spike filter: 1.5x 20-period average volume ---
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(camarilla_r3_12h[i]) or np.isnan(camarilla_s3_12h[i]) or 
            np.isnan(ema_50_1w_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend: price above/below weekly EMA50
        uptrend = close[i] > ema_50_1w_12h[i]
        downtrend = close[i] < ema_50_1w_12h[i]
        
        if position == 0:
            # Look for entries: Camarilla breakout + volume spike + trend alignment
            if uptrend and close[i] > camarilla_r3_12h[i] and volume_spike[i]:
                # Long: uptrend + break above R3 + volume spike
                signals[i] = 0.25
                position = 1
            elif downtrend and close[i] < camarilla_s3_12h[i] and volume_spike[i]:
                # Short: downtrend + break below S3 + volume spike
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: reverse signal or trend change
            if position == 1:
                # Exit long: downtrend OR price breaks below S3 (reversal signal)
                if downtrend or close[i] < camarilla_s3_12h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: uptrend OR price breaks above R3 (reversal signal)
                if uptrend or close[i] > camarilla_r3_12h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals