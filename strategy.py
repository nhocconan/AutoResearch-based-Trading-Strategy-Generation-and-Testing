#!/usr/bin/env python3
"""
6h_1wPivot_FadeWithTrend
Hypothesis: Trade mean-reversion from weekly pivot extremes (R4/S4) when price is 
extended beyond weekly Bollinger Bands, filtered by daily trend direction.
This strategy targets 12-37 trades/year per symbol (50-150 total over 4 years) by:
- Using weekly pivot R4/S4 as extreme support/resistance levels
- Entering mean-reversion trades when price touches these levels with rejection
- Filtering by daily EMA34 trend to align with higher timeframe momentum
- Using volume confirmation to avoid false signals
Works in bull/bear markets by combining mean-reversion at extremes with trend alignment.
"""

name = "6h_1wPivot_FadeWithTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Weekly OHLC for Pivot Points ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points from previous week
    ph_w = df_1w['high'].values
    pl_w = df_1w['low'].values
    pc_w = df_1w['close'].values
    
    # Weekly pivot point and support/resistance levels
    pw_p = (ph_w + pl_w + pc_w) / 3.0
    pw_r4 = pw_p + 3 * (ph_w - pl_w)  # Most extreme resistance
    pw_s4 = pw_p - 3 * (ph_w - pl_w)  # Most extreme support
    
    # Align weekly pivot levels to 6h timeframe
    r4_6h = align_htf_to_ltf(prices, df_1w, pw_r4)
    s4_6h = align_htf_to_ltf(prices, df_1w, pw_s4)
    pivot_6h = align_htf_to_ltf(prices, df_1w, pw_p)
    
    # === Daily Trend Filter (EMA34) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_6h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # === Volume Filter (1.5x 20-period EMA on 6h) ===
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_ok = volume > vol_ema20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (covers weekly/daily calculations)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(r4_6h[i]) or np.isnan(s4_6h[i]) or np.isnan(ema34_6h[i]) or 
            np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long setup: price at S4 rejection with uptrend bias and volume
            if (close[i] <= s4_6h[i] * 1.005 and  # Allow small tolerance for touch
                close[i] > s4_6h[i] and           # Must be above actual S4
                close[i] > ema34_6h[i] and        # Above daily EMA (uptrend bias)
                volume_ok[i]):
                signals[i] = 0.25
                position = 1
            # Short setup: price at R4 rejection with downtrend bias and volume
            elif (close[i] >= r4_6h[i] * 0.995 and  # Allow small tolerance for touch
                  close[i] < r4_6h[i] and           # Must be below actual R4
                  close[i] < ema34_6h[i] and        # Below daily EMA (downtrend bias)
                  volume_ok[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reaches weekly pivot or shows weakness
            if (close[i] >= pivot_6h[i] or  # Reached pivot point
                close[i] < ema34_6h[i]):    # Lost uptrend bias
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price reaches weekly pivot or shows weakness
            if (close[i] <= pivot_6h[i] or  # Reached pivot point
                close[i] > ema34_6h[i]):    # Lost downtrend bias
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals