#!/usr/bin/env python3
# 12h_KAMA_Trend_With_RSI_Filter
# Hypothesis: 12h KAMA identifies adaptive trend direction; RSI(14) filters overbought/oversold extremes.
# Long when: KAMA rising AND RSI < 40 (pullback in uptrend)
# Short when: KAMA falling AND RSI > 60 (bounce in downtrend)
# Exit when: KAMA direction reverses OR RSI returns to neutral zone (40-60)
# Uses 1d ADX(14) as regime filter: only trade when ADX > 20 (trending market)
# Designed for low trade frequency (~15-25/year) to avoid fee drag on 12h timeframe.
# Works in bull by buying dips in uptrend; works in bear by selling bounces in downtrend.

name = "12h_KAMA_Trend_With_RSI_Filter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for ADX filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 12ohlc
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # --- KAMA (Adaptive Moving Average) ---
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)
    # Handle arrays properly
    er = np.zeros(n)
    for i in range(n):
        if i < 10:
            er[i] = np.nan
        else:
            vol_sum = np.sum(np.abs(np.diff(close[i-9:i+1])))
            if vol_sum > 0:
                er[i] = change[i] / vol_sum
            else:
                er[i] = 0
    # Smoothing constants
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # fast=2, slow=30
    kama = np.full(n, np.nan)
    kama[0] = close[0]
    for i in range(1, n):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # KAMA slope (1-period change)
    kama_slope = np.full(n, np.nan)
    for i in range(1, n):
        kama_slope[i] = kama[i] - kama[i-1]
    
    # --- RSI(14) ---
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    rsi = np.full(n, np.nan)
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    
    # Initial averages
    if n >= 14:
        avg_gain[13] = np.mean(gain[0:14])
        avg_loss[13] = np.mean(loss[0:14])
        for i in range(14, n):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
            rs = avg_gain[i] / avg_loss[i] if avg_loss[i] != 0 else 0
            rsi[i] = 100 - (100 / (1 + rs))
    
    # --- 1d ADX(14) for trend regime ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(np.diff(high_1d, prepend=high_1d[0]))
    tr3 = np.abs(np.diff(low_1d, prepend=low_1d[0]))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.insert(dm_plus, 0, 0)
    dm_minus = np.insert(dm_minus, 0, 0)
    
    # Smoothed TR, DM+
    tr_14 = np.full(len(tr_1d), np.nan)
    dm_plus_14 = np.full(len(dm_plus), np.nan)
    dm_minus_14 = np.full(len(dm_minus), np.nan)
    
    if len(tr_1d) >= 14:
        tr_14[13] = np.sum(tr_1d[0:14])
        dm_plus_14[13] = np.sum(dm_plus[0:14])
        dm_minus_14[13] = np.sum(dm_minus[0:14])
        for i in range(14, len(tr_1d)):
            tr_14[i] = tr_14[i-1] - (tr_14[i-1] / 14) + tr_1d[i]
            dm_plus_14[i] = dm_plus_14[i-1] - (dm_plus_14[i-1] / 14) + dm_plus[i]
            dm_minus_14[i] = dm_minus_14[i-1] - (dm_minus_14[i-1] / 14) + dm_minus[i]
    
    # DI+ and DI-
    di_plus = np.full(len(tr_1d), np.nan)
    di_minus = np.full(len(tr_1d), np.nan)
    dx = np.full(len(tr_1d), np.nan)
    
    for i in range(14, len(tr_1d)):
        if tr_14[i] != 0:
            di_plus[i] = 100 * (dm_plus_14[i] / tr_14[i])
            di_minus[i] = 100 * (dm_minus_14[i] / tr_14[i])
            dx[i] = 100 * np.abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])
    
    # ADX = smoothed DX
    adx_1d = np.full(len(tr_1d), np.nan)
    if len(dx) >= 14:
        # First ADX is average of first 14 DX values
        valid_dx = dx[14:28]
        if not np.all(np.isnan(valid_dx)):
            adx_1d[27] = np.nanmean(valid_dx)
            for i in range(28, len(dx)):
                if not np.isnan(dx[i]) and not np.isnan(adx_1d[i-1]):
                    adx_1d[i] = (adx_1d[i-1] * 13 + dx[i]) / 14
    
    # Align 1d indicators to 12h
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max(KAMA needs 1, RSI needs 14, ADX needs ~28)
    start_idx = max(1, 14, 28)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(kama_slope[i]) or
            np.isnan(rsi[i]) or
            np.isnan(adx_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: only trade when ADX > 20 (trending market)
        trending = adx_1d_aligned[i] > 20
        
        if position == 0:
            if trending:
                # Long: KAMA rising AND RSI < 40 (pullback in uptrend)
                if kama_slope[i] > 0 and rsi[i] < 40:
                    signals[i] = 0.25
                    position = 1
                # Short: KAMA falling AND RSI > 60 (bounce in downtrend)
                elif kama_slope[i] < 0 and rsi[i] > 60:
                    signals[i] = -0.25
                    position = -1
        else:
            if position == 1:
                # Exit long: KAMA turns down OR RSI returns to neutral (>50)
                if kama_slope[i] <= 0 or rsi[i] >= 50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: KAMA turns up OR RSI returns to neutral (<50)
                if kama_slope[i] >= 0 or rsi[i] <= 50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals