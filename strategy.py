#!/usr/bin/env python3
# 6H_1W_1D_PivotBreakout_TrendFilter
# Hypothesis: On 6h timeframe, enter long when price breaks above weekly pivot + weekly trend up, short when breaks below weekly pivot + weekly trend down. Uses daily volume confirmation to avoid low-volume breakouts. Weekly pivots provide strong institutional levels, and trend filter avoids counter-trend trades. Designed for 50-150 total trades over 4 years.
# Uses weekly high/low/close from prior week to calculate classic pivot points (P, R1, S1, R2, S2). Entry on break of R1/S1 with trend alignment. Exit on break of opposite level or trend reversal.

name = "6H_1W_1D_PivotBreakout_TrendFilter"
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
    
    # Get weekly data for pivot points and trend
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 2:
        return np.zeros(n)
    
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values
    
    # Calculate weekly pivot points (based on prior week)
    typical_price_w = (high_w + low_w + close_w) / 3
    range_w = high_w - low_w
    # Pivot point (P)
    pivot = typical_price_w
    # Resistance and Support levels
    r1 = 2 * pivot - low_w
    s1 = 2 * pivot - high_w
    r2 = pivot + range_w
    s2 = pivot - range_w
    
    # Weekly trend: EMA(34) on weekly close
    ema_34_w = pd.Series(close_w).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_up_w = close_w > ema_34_w
    
    # Daily volume confirmation: current volume > 1.3x 24-period average (approx 6 days)
    volume_avg = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > (volume_avg * 1.3)
    
    # Align weekly indicators to 6h
    pivot_aligned = align_htf_to_ltf(prices, df_w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_w, s2)
    trend_up_w_aligned = align_htf_to_ltf(prices, df_w, trend_up_w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(trend_up_w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above R1 + weekly uptrend + volume confirmation
            if close[i] > r1_aligned[i] and trend_up_w_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S1 + weekly downtrend + volume confirmation
            elif close[i] < s1_aligned[i] and not trend_up_w_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below S1 (reversal) or trend changes to down
            if close[i] < s1_aligned[i] or not trend_up_w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above R1 (reversal) or trend changes to up
            if close[i] > r1_aligned[i] or trend_up_w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals