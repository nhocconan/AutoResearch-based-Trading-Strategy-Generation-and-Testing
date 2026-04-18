#!/usr/bin/env python3
"""
6h_WeeklyPivot_Breakout_Volume
Hypothesis: Trade 6h breakouts above/below weekly pivot levels (R1/S1) with volume confirmation in the direction of 1w EMA trend. Weekly pivots provide strong institutional support/resistance, breakouts indicate institutional participation, and volume confirms follow-through. EMA filter prevents counter-trend trades. Designed for 10-30 trades/year to avoid fee drag. Works in bull/bear via trend filter.
"""

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
    
    # Get weekly data for pivot points and trend
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly OHLC for pivot calculation
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot points: P = (H+L+C)/3, R1 = 2P - L, S1 = 2P - H
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    
    # Weekly EMA trend filter
    ema_period = 21
    if len(close_1w) >= ema_period:
        ema_1w = np.zeros_like(close_1w)
        ema_1w[ema_period-1] = np.mean(close_1w[:ema_period])
        for i in range(ema_period, len(close_1w)):
            ema_1w[i] = (close_1w[i] * 2 / (ema_period + 1)) + (ema_1w[i-1] * (ema_period - 1) / (ema_period + 1))
    else:
        ema_1w = np.full_like(close_1w, np.nan)
    
    # Align weekly data to 6h
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume confirmation: volume > 1.5x 24-period average (4 days)
    vol_ma = np.zeros_like(volume)
    vol_period = 24
    for i in range(vol_period, len(volume)):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, vol_period, ema_period)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_1w_aligned[i]) or np.isnan(s1_1w_aligned[i]) or 
            np.isnan(ema_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume and above weekly EMA
            if close[i] > r1_1w_aligned[i] and vol_confirm and close[i] > ema_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume and below weekly EMA
            elif close[i] < s1_1w_aligned[i] and vol_confirm and close[i] < ema_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price closes below S1 (reverse signal) or below weekly EMA
            if close[i] < s1_1w_aligned[i] or close[i] < ema_1w_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price closes above R1 (reverse signal) or above weekly EMA
            if close[i] > r1_1w_aligned[i] or close[i] > ema_1w_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_Breakout_Volume"
timeframe = "6h"
leverage = 1.0