#!/usr/bin/env python3
# 6h_PivotReversal_1dTrend_Volume
# Hypothesis: Fade moves to weekly pivot levels (R1/S1) when price is overextended from 1d EMA20,
# with volume confirmation. Works in both bull and bear markets by combining mean reversion
# at key levels with trend filtering and volume validation. Targets 20-40 trades/year.

name = "6h_PivotReversal_1dTrend_Volume"
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
    
    # === 1d EMA20 for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # === Weekly pivot levels (R1, S1) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    wk_high = df_1w['high'].values
    wk_low = df_1w['low'].values
    wk_close = df_1w['close'].values
    
    wk_high_prev = np.roll(wk_high, 1)
    wk_low_prev = np.roll(wk_low, 1)
    wk_close_prev = np.roll(wk_close, 1)
    wk_high_prev[0] = np.nan
    wk_low_prev[0] = np.nan
    wk_close_prev[0] = np.nan
    
    pivot = (wk_high_prev + wk_low_prev + wk_close_prev) / 3.0
    r1 = pivot + (wk_high_prev - wk_low_prev) / 2.0
    s1 = pivot - (wk_high_prev - wk_low_prev) / 2.0
    
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # === Deviation from 1d EMA20 (%) ===
    ema_dev = (close - ema_20_1d_aligned) / ema_20_1d_aligned * 100
    
    # === Volume confirmation (20-period average) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema_20_1d_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_dev[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Conditions
        near_r1 = high[i] >= r1_aligned[i] * 0.998  # Within 0.2% of R1
        near_s1 = low[i] <= s1_aligned[i] * 1.002   # Within 0.2% of S1
        overextended_up = ema_dev[i] > 2.0          # >2% above EMA20
        overextended_down = ema_dev[i] < -2.0       # >2% below EMA20
        vol_ok = volume[i] > vol_ma_20[i]
        
        if position == 0:
            # LONG: price near S1, oversold, volume confirmation
            if near_s1 and overextended_down and vol_ok:
                signals[i] = 0.25
                position = 1
            # SHORT: price near R1, overbought, volume confirmation
            elif near_r1 and overextended_up and vol_ok:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: price crosses above EMA20 or reaches R1
            if close[i] >= ema_20_1d_aligned[i] or high[i] >= r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price crosses below EMA20 or reaches S1
            if close[i] <= ema_20_1d_aligned[i] or low[i] <= s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals