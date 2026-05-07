# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# 1D_Weekly_Pivot_S1_S3_Breakout_1WTrend_Volume_Confirm
# Hypothesis: Weekly pivot levels (S1/S3) act as strong support/resistance in BTC/ETH.
# Breakouts with volume confirmation and weekly trend filter yield high-probability trades.
# Weekly trend avoids whipsaws; volume ensures conviction. Designed for 1d timeframe
# to keep trades < 25/year, reducing fee drag. Works in bull (breakouts) and bear
# (mean reversion at S1/S3 with trend filter) by only trading in trend direction.

name = "1D_Weekly_Pivot_S1_S3_Breakout_1WTrend_Volume_Confirm"
timeframe = "1d"
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
    
    # Get weekly data for pivot calculation and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # Calculate weekly pivot points (S1, S3, R1, R3) from previous week
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly Pivot Point calculation
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    # S1 and S3 levels
    s1_1w = 2.0 * pivot_1w - high_1w
    s3_1w = pivot_1w - 2.0 * (high_1w - low_1w)
    # R1 and R3 levels (for exit)
    r1_1w = 2.0 * pivot_1w - low_1w
    r3_1w = pivot_1w + 2.0 * (high_1w - low_1w)
    
    # Align weekly levels to daily timeframe (use previous week's levels)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    
    # Weekly trend filter: EMA50
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume confirmation: 2.0x average volume (50-period for stability)
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 50)  # Ensure we have EMA50 and volume MA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(s1_1w_aligned[i]) or np.isnan(s3_1w_aligned[i]) or 
            np.isnan(r1_1w_aligned[i]) or np.isnan(r3_1w_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above weekly S1, price above weekly EMA50 (uptrend), volume spike
            if (close[i] > s1_1w_aligned[i] and 
                close[i] > ema50_1w_aligned[i] and 
                volume[i] > 2.0 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S3, price below weekly EMA50 (downtrend), volume spike
            elif (close[i] < s3_1w_aligned[i] and 
                  close[i] < ema50_1w_aligned[i] and 
                  volume[i] > 2.0 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price returns to or below weekly R1 (opposite level)
            if close[i] <= r1_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns to or above weekly R3 (opposite level)
            if close[i] >= r3_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals