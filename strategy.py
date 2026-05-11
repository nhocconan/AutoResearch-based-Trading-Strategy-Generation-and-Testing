#!/usr/bin/env python3
"""
6h_Camarilla_R3_S3_Breakout_1wTrend_Reversal
Hypothesis: Fade at weekly Camarilla R3/S3 levels during strong weekly trends (1w EMA50 slope) with volume confirmation.
In strong uptrends, price often reverses at weekly R3; in strong downtrends, at weekly S3.
Uses 6h timeframe for entry timing, weekly trend for direction filter.
Targets 20-40 trades/year (80-160 total over 4 years) to avoid fee drag.
Works in bull by selling R3 reversals in uptrend; works in bear by buying S3 reversals in downtrend.
"""

name = "6h_Camarilla_R3_S3_Breakout_1wTrend_Reversal"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for Camarilla levels and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # 6h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Weekly typical price (HLC/3) ---
    typical_price_1w = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    
    # --- Weekly Camarilla levels (based on previous week) ---
    # Calculate for each week using prior week's HLC
    high_shift = df_1w['high'].shift(1)
    low_shift = df_1w['low'].shift(1)
    close_shift = df_1w['close'].shift(1)
    typical_price_prev = (high_shift + low_shift + close_shift) / 3
    
    # Camarilla multipliers
    R3 = typical_price_prev + 1.1 * (high_shift - low_shift) * 1.1 / 2
    S3 = typical_price_prev - 1.1 * (high_shift - low_shift) * 1.1 / 2
    R4 = typical_price_prev + 1.1 * (high_shift - low_shift) * 1.1
    S4 = typical_price_prev - 1.1 * (high_shift - low_shift) * 1.1
    
    # --- Weekly EMA50 trend ---
    close_1w = df_1w['close'].values
    ema_1w = np.full(len(close_1w), np.nan)
    for i in range(len(close_1w)):
        if i < 50:
            ema_1w[i] = np.nan
        elif i == 50:
            ema_1w[i] = np.mean(close_1w[0:50])
        else:
            ema_1w[i] = (close_1w[i] * 2 / (50 + 1)) + (ema_1w[i-1] * (49 / (50 + 1)))
    
    # EMA slope
    ema_slope_1w = np.full(len(close_1w), np.nan)
    for i in range(51, len(close_1w)):
        ema_slope_1w[i] = ema_1w[i] - ema_1w[i-1]
    
    # --- 6h ATR(14) for volatility scaling ---
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
    
    # --- 6h volume MA(20) ---
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Align weekly indicators to 6h
    R3_1w = align_htf_to_ltf(prices, df_1w, R3.values)
    S3_1w = align_htf_to_ltf(prices, df_1w, S3.values)
    R4_1w = align_htf_to_ltf(prices, df_1w, R4.values)
    S4_1w = align_htf_to_ltf(prices, df_1w, S4.values)
    ema_slope_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_slope_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max(weekly data needs 1 week, EMA50, ATR14, vol MA20)
    start_idx = max(50, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(R3_1w[i]) or
            np.isnan(S3_1w[i]) or
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
            # Fade at R3/S3 in strong weekly trends
            if close[i] < R3_1w[i] and ema_slope_1w_aligned[i] > 0 and vol_spike:
                # Short at R3 resistance in uptrend
                signals[i] = -0.25
                position = -1
            elif close[i] > S3_1w[i] and ema_slope_1w_aligned[i] < 0 and vol_spike:
                # Long at S3 support in downtrend
                signals[i] = 0.25
                position = 1
        else:
            if position == 1:
                # Exit long: price crosses S3 OR weekly trend turns up
                if close[i] < S3_1w[i] or ema_slope_1w_aligned[i] > 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price crosses R3 OR weekly trend turns down
                if close[i] > R3_1w[i] or ema_slope_1w_aligned[i] < 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals