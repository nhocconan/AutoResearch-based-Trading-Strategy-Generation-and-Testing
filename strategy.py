#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_1w_Pivot_R3S3_MomentumBreakout"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 150:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 30 or len(df_1w) < 12:
        return np.zeros(n)
    
    # === Daily Pivot Points (previous day) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values for pivot calculation
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    
    # Pivot point
    pivot = (prev_high + prev_low + prev_close) / 3
    range_val = prev_high - prev_low
    
    # Key levels: R3 and S3 (wider bands for 6h timeframe)
    r3 = pivot + (range_val * 1.1 * 1.1)  # R3 = pivot + 1.1 * range
    s3 = pivot - (range_val * 1.1 * 1.1)  # S3 = pivot - 1.1 * range
    
    # Align to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    
    # === Weekly Trend Filter ===
    # Weekly close trend: above/below weekly open
    weekly_open = df_1w['open'].values
    weekly_close = df_1w['close'].values
    weekly_bullish = weekly_close > weekly_open
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    
    # === 6h Momentum Confirmation ===
    close_series = pd.Series(prices['close'].values)
    roc_6 = close_series.pct_change(periods=6).values  # 6-period ROC (36 hours)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get values
        close_val = prices['close'].iloc[i]
        roc_val = roc_6[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        pivot_val = pivot_aligned[i]
        weekly_bull = weekly_bullish_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(roc_val) or np.isnan(r3_val) or 
            np.isnan(s3_val) or np.isnan(pivot_val) or np.isnan(weekly_bull)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above R3 with upward momentum and weekly bullish bias
            if close_val > r3_val and roc_val > 0.005 and weekly_bull > 0.5:
                signals[i] = 0.25
                position = 1
            # Short: Break below S3 with downward momentum and weekly bearish bias
            elif close_val < s3_val and roc_val < -0.005 and weekly_bull < 0.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns below pivot or momentum turns negative
            if close_val < pivot_val or roc_val < -0.003:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns above pivot or momentum turns positive
            if close_val > pivot_val or roc_val > 0.003:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals