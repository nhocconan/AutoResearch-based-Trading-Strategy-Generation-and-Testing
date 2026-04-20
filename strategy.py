#!/usr/bin/env python3
# 6h_1w_WeeklyPivot_Breakout_VolumeTrend
# Hypothesis: Breakouts of weekly pivot levels (R1/S1) with volume confirmation and 1d EMA trend filter.
# Weekly pivots capture longer-term structure; volume confirms institutional interest; EMA filter avoids counter-trend trades.
# Works in bull/bear via trend filter. Target: 15-35 trades per year per symbol.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_WeeklyPivot_Breakout_VolumeTrend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop for pivots
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points (standard formula)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    range_1w = high_1w - low_1w
    
    # Weekly R1 and S1 (commonly used levels)
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    
    # Get daily data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all weekly and daily levels to 6h
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 6h: Volume ratio (current vs 24-period average)
    volume = prices['volume'].values
    vol_ma24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_ratio = volume / np.where(vol_ma24 > 0, vol_ma24, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after EMA and volume MA warmup
        # Get values
        close_val = prices['close'].iloc[i]
        r1_1w_val = r1_1w_aligned[i]
        s1_1w_val = s1_1w_aligned[i]
        ema50_1d_val = ema50_1d_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(r1_1w_val) or np.isnan(s1_1w_val) or np.isnan(ema50_1d_val) or 
            np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above weekly R1 with volume confirmation and above daily EMA50
            if (close_val > r1_1w_val and vol_ratio_val > 2.0 and 
                close_val > ema50_1d_val):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below weekly S1 with volume confirmation and below daily EMA50
            elif (close_val < s1_1w_val and vol_ratio_val > 2.0 and 
                  close_val < ema50_1d_val):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns to or below weekly S1 (opposite level)
            if close_val <= s1_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns to or above weekly R1 (opposite level)
            if close_val >= r1_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals