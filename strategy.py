#!/usr/bin/env python3
"""
1d_WeeklyPivot_Breakout_1wTrend
Hypothesis: Price breaks the weekly pivot resistance (long) or support (short) calculated from weekly data, with weekly EMA20 trend filter and volume confirmation.
Breakouts from weekly pivot levels capture significant market turning points, while weekly trend filter ensures alignment with longer-term direction.
Volume confirmation filters false breakouts. Works in bull/bear by trading only in direction of weekly trend.
Target: 15-25 trades/year (60-100 total) to minimize fee drag.
"""

name = "1d_WeeklyPivot_Breakout_1wTrend"
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
    
    # Weekly data for pivot points and trend
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Weekly pivot points (using prior week's data)
    pivot_1w = np.full(len(high_1w), np.nan)
    r1_1w = np.full(len(high_1w), np.nan)
    s1_1w = np.full(len(high_1w), np.nan)
    
    if len(high_1w) >= 2:
        for i in range(1, len(high_1w)):
            pivot_1w[i] = (high_1w[i-1] + low_1w[i-1] + close_1w[i-1]) / 3.0
            r1_1w[i] = 2 * pivot_1w[i] - low_1w[i-1]
            s1_1w[i] = 2 * pivot_1w[i] - high_1w[i-1]
    
    # Weekly EMA20 for trend filter
    ema20_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 20:
        ema20_1w[19] = np.mean(close_1w[:20])
        alpha = 2 / (20 + 1)
        for i in range(20, len(close_1w)):
            ema20_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema20_1w[i-1]
    
    # Weekly volume SMA10 for volume confirmation
    vol_sma10_1w = np.full(len(volume_1w), np.nan)
    if len(volume_1w) >= 10:
        vol_sma10_1w[9] = np.mean(volume_1w[:10])
        for i in range(10, len(volume_1w)):
            vol_sma10_1w[i] = (vol_sma10_1w[i-1] * 9 + volume_1w[i]) / 10
    
    # Align weekly indicators to daily
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    vol_sma10_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_sma10_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for EMA20
    
    for i in range(start_idx, n):
        if np.isnan(pivot_1w_aligned[i]) or np.isnan(r1_1w_aligned[i]) or np.isnan(s1_1w_aligned[i]) or np.isnan(ema20_1w_aligned[i]) or np.isnan(vol_sma10_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current daily volume > 1.5x average weekly volume (scaled)
        # 5x daily bars in weekly (since 1w is 5x 1d)
        vol_1w_scaled = vol_sma10_1w_aligned[i] / 5.0  # Average daily-equivalent volume from weekly data
        volume_confirm = volume[i] > 1.5 * vol_1w_scaled
        
        # Trend and price relative to weekly pivot levels
        is_uptrend = close[i] > ema20_1w_aligned[i]
        is_downtrend = close[i] < ema20_1w_aligned[i]
        price_above_r1 = close[i] > r1_1w_aligned[i]
        price_below_s1 = close[i] < s1_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above weekly R1, in uptrend, with volume
            if price_above_r1 and is_uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S1, in downtrend, with volume
            elif price_below_s1 and is_downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price falls back below weekly pivot or trend turns down
            if close[i] < pivot_1w_aligned[i] or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price rises back above weekly pivot or trend turns up
            if close[i] > pivot_1w_aligned[i] or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals