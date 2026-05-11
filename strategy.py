#!/usr/bin/env python3
"""
1d_1w_KAMA_Trend_With_Volume_And_Chop_Filter
Hypothesis: On daily timeframe, use Kaufman Adaptive Moving Average (KAMA) to detect trend direction,
combined with volume confirmation and Choppiness Index to filter ranging markets.
In bull markets: go long when KAMA turns up, volume expands, and market is trending (low chop).
In bear markets: go short when KAMA turns down, volume expands, and market is trending.
Uses weekly timeframe for trend filter to avoid counter-trend trades.
Target: 15-25 trades/year (60-100 total over 4 years) to minimize fee drag.
Works in both bull and bear by adapting to trend via KAMA and avoiding choppy markets.
"""

name = "1d_1w_KAMA_Trend_With_Volume_And_Chop_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Daily OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Daily KAMA (10, 2, 30) ---
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # 10-period volatility
    er = np.where(volatility != 0, change / volatility, 0)
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    
    # KAMA calculation
    kama = np.full(n, np.nan)
    kama[9] = close[9]  # start after first 10 bars
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # KAMA slope (trend direction)
    kama_slope = np.full(n, np.nan)
    kama_slope[10:] = np.diff(kama, n=1)[9:]  # align with kama
    
    # --- Daily Volume MA(20) ---
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # --- Daily Choppiness Index (14) ---
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Sum of TR over 14 periods
    tr_sum = np.full(n, np.nan)
    for i in range(13, n):
        if i == 13:
            tr_sum[i] = np.sum(tr[0:14])
        else:
            tr_sum[i] = tr_sum[i-1] + tr[i] - tr[i-14]
    
    # Highest high and lowest low over 14 periods
    max_high = np.full(n, np.nan)
    min_low = np.full(n, np.nan)
    for i in range(13, n):
        if i == 13:
            max_high[i] = np.max(high[0:14])
            min_low[i] = np.min(low[0:14])
        else:
            max_high[i] = max(max_high[i-1], high[i])
            min_low[i] = min(min_low[i-1], low[i])
    
    # Chop calculation
    chop = np.full(n, np.nan)
    for i in range(13, n):
        if tr_sum[i] > 0 and (max_high[i] - min_low[i]) > 0:
            chop[i] = 100 * np.log10(tr_sum[i] / (max_high[i] - min_low[i])) / np.log10(14)
        else:
            chop[i] = 50  # neutral if undefined
    
    # --- Weekly EMA(10) for trend filter ---
    close_1w = df_1w['close'].values
    ema_1w = np.full(len(close_1w), np.nan)
    for i in range(len(close_1w)):
        if i < 10:
            ema_1w[i] = np.nan
        elif i == 10:
            ema_1w[i] = np.mean(close_1w[0:10])
        else:
            ema_1w[i] = (close_1w[i] * 2 / (10 + 1)) + (ema_1w[i-1] * (9 / (10 + 1)))
    
    # Align daily indicators (already aligned to daily index)
    # Align weekly EMA to daily
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max(KAMA needs 10, Vol MA needs 20, Chop needs 13, EMA needs 10)
    start_idx = max(10, 20, 13, 10)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(kama[i]) or
            np.isnan(kama_slope[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(chop[i]) or
            np.isnan(ema_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Conditions
        kama_up = kama_slope[i] > 0
        kama_down = kama_slope[i] < 0
        vol_spike = volume[i] > vol_ma[i] * 1.5
        trending = chop[i] < 38.2  # trending market (low chop)
        weekly_uptrend = close[i] > ema_1w_aligned[i]
        weekly_downtrend = close[i] < ema_1w_aligned[i]
        
        if position == 0:
            if kama_up and vol_spike and trending and weekly_uptrend:
                # Long: KAMA turning up with volume in uptrend
                signals[i] = 0.25
                position = 1
            elif kama_down and vol_spike and trending and weekly_downtrend:
                # Short: KAMA turning down with volume in downtrend
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: KAMA turns down OR weekly trend turns down
                if kama_down or not weekly_uptrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: KAMA turns up OR weekly trend turns up
                if kama_up or not weekly_downtrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals