#!/usr/bin/env python3
"""
6h_Weekly_Pivot_Reversal_With_Volume
Hypothesis: Weekly pivots act as strong support/resistance in ranging markets (2025+), while volume spikes confirm institutional interest at these levels. In trending markets, price respects the weekly pivot as dynamic S/R. This strategy takes mean-reversion entries at weekly R2/S2 with volume confirmation, filtered by 1d trend (EMA50) to avoid counter-trend trades. Designed for 6H timeframe to balance trade frequency (~20-40/year) and capture both mean-reversion and trend continuation at key weekly levels.
"""

name = "6h_Weekly_Pivot_Reversal_With_Volume"
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
    
    # Volume spike: >1.8x 24-period average (6h * 4 = 1 day)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    # Weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points (standard formula)
    # Pivot = (H + L + C) / 3
    # R2 = Pivot + (High - Low)
    # S2 = Pivot - (High - Low)
    pivot = (high_1w + low_1w + close_1w) / 3
    weekly_range = high_1w - low_1w
    weekly_r2 = pivot + weekly_range
    weekly_s2 = pivot - weekly_range
    
    # 1d trend filter: EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all indicators to 6h timeframe
    weekly_r2_aligned = align_htf_to_ltf(prices, df_1w, weekly_r2)
    weekly_s2_aligned = align_htf_to_ltf(prices, df_1w, weekly_s2)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        if (np.isnan(weekly_r2_aligned[i]) or
            np.isnan(weekly_s2_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price at weekly S2 with volume spike and above 1d EMA50 (bullish alignment)
            if (close[i] <= weekly_s2_aligned[i] * 1.005 and  # Allow 0.5% tolerance
                volume_spike[i] and
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price at weekly R2 with volume spike and below 1d EMA50 (bearish alignment)
            elif (close[i] >= weekly_r2_aligned[i] * 0.995 and  # Allow 0.5% tolerance
                  volume_spike[i] and
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches weekly pivot OR breaks below weekly S2
            if (close[i] >= ((weekly_s2_aligned[i] + pivot) / 2) or  # Midpoint between S2 and pivot
                close[i] < weekly_s2_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches weekly pivot OR breaks above weekly R2
            if (close[i] <= ((weekly_r2_aligned[i] + pivot) / 2) or  # Midpoint between R2 and pivot
                close[i] > weekly_r2_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals