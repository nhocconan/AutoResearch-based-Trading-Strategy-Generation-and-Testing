#!/usr/bin/env python3
"""
6h_WeeklyPivot_Trend_Breakout
Hypothesis: Price breaks above weekly R4 (long) or below weekly S4 (short) with weekly EMA50 trend filter and volume confirmation.
Weekly pivots capture key support/resistance from longer timeframe, while weekly trend filter ensures alignment.
Works in bull/bear by trading only in direction of weekly trend. Target: 15-30 trades/year (60-120 total) to minimize fee drag.
"""

name = "6h_WeeklyPivot_Trend_Breakout"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
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
    
    # Calculate weekly pivot points (standard formula)
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    # R4 = R3 + (H - L), S4 = S3 - (H - L)
    pivot_1w = np.full(len(high_1w), np.nan)
    r4_1w = np.full(len(high_1w), np.nan)
    s4_1w = np.full(len(high_1w), np.nan)
    
    if len(high_1w) >= 1:
        for i in range(len(high_1w)):
            if not (np.isnan(high_1w[i]) or np.isnan(low_1w[i]) or np.isnan(close_1w[i])):
                pivot = (high_1w[i] + low_1w[i] + close_1w[i]) / 3.0
                hl_range = high_1w[i] - low_1w[i]
                r4_1w[i] = pivot + 2 * hl_range  # R4 = R3 + (H-L) where R3 = H + 2*(P-L) = P + 2*(H-L)
                s4_1w[i] = pivot - 2 * hl_range  # S4 = S3 - (H-L) where S3 = L - 2*(H-P) = P - 2*(H-L)
    
    # Weekly EMA50 for trend filter
    ema50_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 50:
        ema50_1w[49] = np.mean(close_1w[:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_1w)):
            ema50_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema50_1w[i-1]
    
    # Weekly volume SMA20 for volume confirmation
    vol_sma20_1w = np.full(len(volume_1w), np.nan)
    if len(volume_1w) >= 20:
        vol_sma20_1w[19] = np.mean(volume_1w[:20])
        for i in range(20, len(volume_1w)):
            vol_sma20_1w[i] = (vol_sma20_1w[i-1] * 19 + volume_1w[i]) / 20
    
    # Align weekly indicators to 6h
    r4_1w_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    s4_1w_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    vol_sma20_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_sma20_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for EMA50
    
    for i in range(start_idx, n):
        if np.isnan(r4_1w_aligned[i]) or np.isnan(s4_1w_aligned[i]) or np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_sma20_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x average weekly volume (scaled)
        # 1w = 28 x 6h bars (7 days * 4 bars/day), so scale weekly volume to 6h equivalent
        vol_1w_scaled = vol_sma20_1w_aligned[i] / 28.0  # Average 6h-equivalent volume from weekly data
        volume_confirm = volume[i] > 1.5 * vol_1w_scaled
        
        # Trend and price relative to weekly pivot levels
        is_uptrend = close[i] > ema50_1w_aligned[i]
        is_downtrend = close[i] < ema50_1w_aligned[i]
        price_above_r4 = close[i] > r4_1w_aligned[i]
        price_below_s4 = close[i] < s4_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above weekly R4, in uptrend, with volume
            if price_above_r4 and is_uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S4, in downtrend, with volume
            elif price_below_s4 and is_downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price falls back below weekly R4 or trend turns down
            if not price_above_r4 or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price rises back above weekly S4 or trend turns up
            if not price_below_s4 or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals