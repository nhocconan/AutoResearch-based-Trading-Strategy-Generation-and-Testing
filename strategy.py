#!/usr/bin/env python3
"""
6h_ADX_WeeklyPivot_Fade_Trend
Hypothesis: In 6h timeframe, price tends to reverse from weekly R3/S3 levels when ADX(14) < 25 (weak trend/range), but continues through R4/S4 when ADX(14) > 25 (strong trend). Weekly pivot levels provide institutional support/resistance, while ADX filters for regime appropriateness. Works in bull/bear by adapting to trend strength.
Target: 12-30 trades/year (50-120 total over 4 years) to minimize fee drag.
"""

name = "6h_ADX_WeeklyPivot_Fade_Trend"
timeframe = "6h"
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
    
    # Weekly data for pivot points
    df_w = get_htf_data(prices, '1w')
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values
    
    # Calculate weekly pivot points (standard floor trader method)
    pivot_w = (high_w + low_w + close_w) / 3.0
    r1_w = 2 * pivot_w - low_w
    s1_w = 2 * pivot_w - high_w
    r2_w = pivot_w + (high_w - low_w)
    s2_w = pivot_w - (high_w - low_w)
    r3_w = high_w + 2 * (pivot_w - low_w)
    s3_w = low_w - 2 * (high_w - pivot_w)
    r4_w = r3_w + (high_w - low_w)
    s4_w = s3_w - (high_w - low_w)
    
    # Daily data for ADX calculation
    df_d = get_htf_data(prices, '1d')
    high_d = df_d['high'].values
    low_d = df_d['low'].values
    close_d = df_d['close'].values
    
    # Calculate ADX(14) on daily data
    period = 14
    # True Range
    tr1 = high_d[1:] - low_d[1:]
    tr2 = np.abs(high_d[1:] - close_d[:-1])
    tr3 = np.abs(low_d[1:] - close_d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align with index
    
    # Directional Movement
    up_move = high_d[1:] - high_d[:-1]
    down_move = low_d[:-1] - low_d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed values
    def smooth_wilder(arr, period):
        smoothed = np.full_like(arr, np.nan)
        if len(arr) < period:
            return smoothed
        smoothed[period-1] = np.nansum(arr[:period])
        for i in range(period, len(arr)):
            if not np.isnan(smoothed[i-1]):
                smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + arr[i]
        return smoothed
    
    tr_smoothed = smooth_wilder(tr, period)
    plus_dm_smoothed = smooth_wilder(plus_dm, period)
    minus_dm_smoothed = smooth_wilder(minus_dm, period)
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smoothed / tr_smoothed
    minus_di = 100 * minus_dm_smoothed / tr_smoothed
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = smooth_wilder(dx, period)
    
    # Align weekly pivot levels to 6h
    pivot_w_aligned = align_htf_to_ltf(prices, df_w, pivot_w)
    r3_w_aligned = align_htf_to_ltf(prices, df_w, r3_w)
    s3_w_aligned = align_htf_to_ltf(prices, df_w, s3_w)
    r4_w_aligned = align_htf_to_ltf(prices, df_w, r4_w)
    s4_w_aligned = align_htf_to_ltf(prices, df_w, s4_w)
    
    # Align daily ADX to 6h
    adx_aligned = align_htf_to_ltf(prices, df_d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for ADX calculation
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_w_aligned[i]) or np.isnan(r3_w_aligned[i]) or np.isnan(s3_w_aligned[i]) or
            np.isnan(r4_w_aligned[i]) or np.isnan(s4_w_aligned[i]) or np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        adx_val = adx_aligned[i]
        price = close[i]
        
        # Regime detection: ADX < 25 = range/weak trend (fade at R3/S3), ADX > 25 = strong trend (breakout at R4/S4)
        is_range = adx_val < 25
        is_trend = adx_val > 25
        
        if position == 0:
            # Fade at R3/S3 in ranging markets
            if is_range:
                if price >= r3_w_aligned[i]:
                    signals[i] = -0.25  # Short at R3 resistance
                    position = -1
                elif price <= s3_w_aligned[i]:
                    signals[i] = 0.25   # Long at S3 support
                    position = 1
            # Breakout continuation at R4/S4 in trending markets
            elif is_trend:
                if price > r4_w_aligned[i]:
                    signals[i] = 0.25   # Long breakout above R4
                    position = 1
                elif price < s4_w_aligned[i]:
                    signals[i] = -0.25  # Short breakdown below S4
                    position = -1
        elif position == 1:
            # Long exit: price falls back below pivot or ADX drops indicating trend weakness
            if price < pivot_w_aligned[i] or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises back above pivot or ADX drops indicating trend weakness
            if price > pivot_w_aligned[i] or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals