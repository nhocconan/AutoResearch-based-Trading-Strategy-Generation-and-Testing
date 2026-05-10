#!/usr/bin/env python3
"""
1d_WeeklyPivot_Breakout_1wTrend
Hypothesis: Price breaks above/below weekly pivot-derived R1/S1 levels with 1-week EMA13 trend filter and volume confirmation.
Weekly pivot acts as strong weekly support/resistance; breakouts with volume and trend alignment capture directional moves.
Works in bull/bear by filtering trades in direction of weekly trend.
Target: 10-25 trades/year (40-100 total) to minimize fee drag.
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
    
    # Weekly data (1w)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Weekly pivot-based levels from prior week: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    pivot_high = high_1w
    pivot_low = low_1w
    pivot_close = close_1w
    weekly_range = pivot_high - pivot_low
    
    r1_level = pivot_close + 1.1 * weekly_range / 12
    s1_level = pivot_close - 1.1 * weekly_range / 12
    
    # Weekly EMA13 for trend filter
    ema13_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 13:
        ema13_1w[12] = np.mean(close_1w[:13])
        alpha = 2 / (13 + 1)
        for i in range(13, len(close_1w)):
            ema13_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema13_1w[i-1]
    
    # Weekly volume SMA5 for volume confirmation (scaled to daily)
    vol_sma5_1w = np.full(len(volume_1w), np.nan)
    if len(volume_1w) >= 5:
        vol_sma5_1w[4] = np.mean(volume_1w[:5])
        for i in range(5, len(volume_1w)):
            vol_sma5_1w[i] = (vol_sma5_1w[i-1] * 4 + volume_1w[i]) / 5
    
    # Align weekly indicators to daily
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1_level)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1_level)
    ema13_1w_aligned = align_htf_to_ltf(prices, df_1w, ema13_1w)
    vol_sma5_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_sma5_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 13  # Wait for EMA13
    
    for i in range(start_idx, n):
        if np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(ema13_1w_aligned[i]) or np.isnan(vol_sma5_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current daily volume > 1.5x average weekly volume (scaled)
        vol_weekly_scaled = vol_sma5_1w_aligned[i] / 5.0  # 5 daily bars in 1w
        volume_confirm = volume[i] > 1.5 * vol_weekly_scaled
        
        # Trend and price relative to weekly levels
        is_uptrend = close[i] > ema13_1w_aligned[i]
        is_downtrend = close[i] < ema13_1w_aligned[i]
        price_above_r1 = close[i] > r1_aligned[i]
        price_below_s1 = close[i] < s1_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1, in uptrend, with volume
            if price_above_r1 and is_uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1, in downtrend, with volume
            elif price_below_s1 and is_downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price falls back below R1 or trend turns down
            if not price_above_r1 or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price rises back above S1 or trend turns up
            if not price_below_s1 or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals