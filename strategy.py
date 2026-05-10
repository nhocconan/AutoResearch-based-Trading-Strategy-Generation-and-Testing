#!/usr/bin/env python3
"""
4h_Camarilla_Pivot_Reversal_12hTrend
Hypothesis: Camarilla pivot reversals (S3/R3) with 12h EMA trend filter and volume confirmation.
Works in bull/bear by taking reversals at strong intraday levels aligned with higher timeframe trend.
Designed for 20-40 trades/year to avoid fee drag while capturing high-probability mean-reversion bounces.
"""

name = "4h_Camarilla_Pivot_Reversal_12hTrend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 12h data for Camarilla pivot calculation and trend filter
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels (using prior 12h bar)
    high_prev = df_12h['high'].shift(1).values
    low_prev = df_12h['low'].shift(1).values
    close_prev = df_12h['close'].shift(1).values
    
    # Camarilla equations: 
    # Pivot = (H+L+C)/3
    # R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), R2 = C + ((H-L)*1.1/6), R1 = C + ((H-L)*1.1/12)
    # S1 = C - ((H-L)*1.1/12), S2 = C - ((H-L)*1.1/6), S3 = C - ((H-L)*1.1/4), S4 = C - ((H-L)*1.1/2)
    pivot = (high_prev + low_prev + close_prev) / 3
    range_val = high_prev - low_prev
    camarilla_r3 = close_prev + (range_val * 1.1 / 4)
    camarilla_s3 = close_prev - (range_val * 1.1 / 4)
    camarilla_r4 = close_prev + (range_val * 1.1 / 2)
    camarilla_s4 = close_prev - (range_val * 1.1 / 2)
    
    # Align Camarilla levels to 4h timeframe (wait for 12h bar to close)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s4)
    pivot_aligned = align_htf_to_ltf(prices, df_12h, pivot)
    
    # 12h EMA50 for trend filter
    ema_50 = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Get 4h price and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 1.8x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema20 * 1.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 12h pivot (needs 1 bar), EMA50 (50 bars), volume EMA (20)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long reversal: price touches/below S3 with volume, in uptrend (above EMA50)
            if low[i] <= camarilla_s3_aligned[i] and close[i] > ema_50_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short reversal: price touches/above R3 with volume, in downtrend (below EMA50)
            elif high[i] >= camarilla_r3_aligned[i] and close[i] < ema_50_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reaches S4 (strong support) or trend turns bearish
            if low[i] <= camarilla_s4_aligned[i] or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reaches R4 (strong resistance) or trend turns bullish
            if high[i] >= camarilla_r4_aligned[i] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals