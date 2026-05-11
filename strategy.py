#!/usr/bin/env python3
"""
4h_1d_Pivot_Reversion_Bias
Hypothesis: In crypto, price tends to revert to daily pivot points (support/resistance) 
with a trend bias from higher timeframe. Long when price touches daily S1 in uptrend 
(1w EMA50 up), short when touches R1 in downtrend (1w EMA50 down). Uses volume 
confirmation to avoid false touches. Works in bull (buy dips to S1) and bear (sell 
rallies to R1). Target: 20-40 trades/year (80-160 total) to minimize fee drag.
"""
name = "4h_1d_Pivot_Reversion_Bias"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1d Classic Pivot Points ---
    # P = (H + L + C)/3
    # R1 = 2*P - L
    # S1 = 2*P - H
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = 2 * pivot_1d - low_1d
    s1_1d = 2 * pivot_1d - high_1d
    
    # --- 1w EMA50 trend filter ---
    close_1w = df_1w['close'].values
    ema_1w = np.full(len(close_1w), np.nan)
    for i in range(len(close_1w)):
        if i < 50:
            ema_1w[i] = np.nan
        elif i == 50:
            ema_1w[i] = np.mean(close_1w[0:50])
        else:
            ema_1w[i] = (close_1w[i] * 2 / (50 + 1)) + (ema_1w[i-1] * (49 / (50 + 1)))
    
    # EMA slope (trend direction)
    ema_slope_1w = np.full(len(close_1w), np.nan)
    for i in range(51, len(close_1w)):
        ema_slope_1w[i] = ema_1w[i] - ema_1w[i-1]
    
    # --- 4h ATR(14) for stop/re-entry logic ---
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = np.full(n, np.nan)
    for i in range(14, n):
        if i == 14:
            atr[i] = np.mean(tr[0:14])
        else:
            atr[i] = (tr[i] * 1 / 14) + (atr[i-1] * 13 / 14)
    
    # --- 4h volume MA(20) for confirmation ---
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Align 1d pivot levels to 4h
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Align 1w indicators to 4h
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    ema_slope_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_slope_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 1d pivot (1 bar), 1w EMA50, ATR14, vol MA20
    start_idx = max(1, 50, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(pivot_1d_aligned[i]) or
            np.isnan(r1_1d_aligned[i]) or
            np.isnan(s1_1d_aligned[i]) or
            np.isnan(ema_1w_aligned[i]) or
            np.isnan(ema_slope_1w_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > vol_ma[i] * 1.5
        
        if position == 0:
            # Long: price touches S1 support in uptrend
            if (low[i] <= s1_1d_aligned[i] and 
                ema_slope_1w_aligned[i] > 0 and 
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Short: price touches R1 resistance in downtrend
            elif (high[i] >= r1_1d_aligned[i] and 
                  ema_slope_1w_aligned[i] < 0 and 
                  vol_spike):
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: price crosses pivot OR trend turns down
                if (high[i] >= pivot_1d_aligned[i] or 
                    ema_slope_1w_aligned[i] < 0):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price crosses pivot OR trend turns up
                if (low[i] <= pivot_1d_aligned[i] or 
                    ema_slope_1w_aligned[i] > 0):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals