#!/usr/bin/env python3
"""
6h_WeeklyPivot_Retest_1dTrend
Strategy: Retest of weekly pivot levels (R1/S1) with 1d trend filter and volume confirmation.
Long: Price retests and fails below weekly S1, then closes back above it in 1d uptrend.
Short: Price retests and fails above weekly R1, then closes back below it in 1d downtrend.
Designed for 6h timeframe: ~15-25 trades/year per symbol (60-100 total over 4 years).
Works in bull/bear via trend filter and mean-reversion retest logic.
"""

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
    
    # Get weekly data for pivot points (using daily data to calculate weekly pivots)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points from weekly OHLC
    high_w = df_1w['high'].values
    low_w = df_1w['low'].values
    close_w = df_1w['close'].values
    
    # Weekly pivot: P = (H + L + C) / 3
    pivot_w = (high_w + low_w + close_w) / 3.0
    # Weekly R1 = 2*P - L
    r1_w = 2 * pivot_w - low_w
    # Weekly S1 = 2*P - H
    s1_w = 2 * pivot_w - high_w
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    close_1d = df_1d['close'].values
    
    # Daily EMA50 and EMA200 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Daily volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all weekly and daily data to 6h timeframe
    r1_w_aligned = align_htf_to_ltf(prices, df_1w, r1_w)
    s1_w_aligned = align_htf_to_ltf(prices, df_1w, s1_w)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need enough for EMA200
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_w_aligned[i]) or np.isnan(s1_w_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(ema_200_aligned[i]) or
            np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend conditions
        uptrend = ema_50_aligned[i] > ema_200_aligned[i]
        downtrend = ema_50_aligned[i] < ema_200_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma_aligned[i]
        
        # Weekly pivot retest conditions
        # Long: price tested below weekly S1, then closed back above it
        retest_long = low[i] < s1_w_aligned[i] and close[i] > s1_w_aligned[i]
        # Short: price tested above weekly R1, then closed back below it
        retest_short = high[i] > r1_w_aligned[i] and close[i] < r1_w_aligned[i]
        
        if position == 0:
            # Long: uptrend + volume + retest long setup
            if uptrend and vol_confirm and retest_long:
                signals[i] = 0.25
                position = 1
            # Short: downtrend + volume + retest short setup
            elif downtrend and vol_confirm and retest_short:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend change, volume confirmation, or retest short setup
            if not uptrend or vol_confirm or retest_short:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend change, volume confirmation, or retest long setup
            if not downtrend or vol_confirm or retest_long:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_Retest_1dTrend"
timeframe = "6h"
leverage = 1.0