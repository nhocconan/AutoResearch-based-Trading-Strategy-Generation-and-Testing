#!/usr/bin/env python3
"""
12h_1D_Camarilla_R3_S3_Breakout_TrendFilter
Hypothesis: Price breaks Camarilla R3/S3 levels from prior 1-day candle, filtered by 1-day EMA34 trend and volume spike.
Long when: price > R3 (bullish breakout) AND 1-day EMA34 rising AND volume > 1.5x 20-bar average.
Short when: price < S3 (bearish breakdown) AND 1-day EMA34 falling AND volume > 1.5x 20-bar average.
Exit when: price returns to Camarilla PIVOT level or EMA34 trend reverses.
Uses 12h timeframe with 1-day HTF for structure. Designed for fewer trades (target: 20-40/year) to avoid fee drag.
Works in bull by buying breakouts in uptrend; works in bear by selling breakdowns in downtrend.
"""

name = "12h_1D_Camarilla_R3_S3_Breakout_TrendFilter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for Camarilla levels and EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 12h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1-day Camarilla levels (based on prior day's range) ---
    # Calculate for each 1d bar: H, L, C from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Shift by 1 to use prior day's values for today's levels
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan  # first day has no prior
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Camarilla formulas
    rang = prev_high - prev_low
    camarilla_pivot = (prev_high + prev_low + prev_close) / 3
    camarilla_r3 = camarilla_pivot + (rang * 1.1 / 4)
    camarilla_s3 = camarilla_pivot - (rang * 1.1 / 4)
    
    # --- 1-day EMA34 trend ---
    ema_34 = np.full(len(close_1d), np.nan)
    for i in range(len(close_1d)):
        if i < 34:
            ema_34[i] = np.nan
        elif i == 34:
            ema_34[i] = np.mean(close_1d[0:34])
        else:
            ema_34[i] = (close_1d[i] * 2 / (34 + 1)) + (ema_34[i-1] * (33 / (34 + 1)))
    
    # EMA slope
    ema_slope_34 = np.full(len(close_1d), np.nan)
    for i in range(35, len(close_1d)):
        ema_slope_34[i] = ema_34[i] - ema_34[i-1]
    
    # --- 12h volume MA(20) ---
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Align 1d indicators to 12h
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    ema_slope_34_aligned = align_htf_to_ltf(prices, df_1d, ema_slope_34)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need at least 2 days for Camarilla (yesterday's data), EMA34, vol MA20
    start_idx = max(2, 35, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(camarilla_pivot_aligned[i]) or
            np.isnan(ema_34_aligned[i]) or
            np.isnan(ema_slope_34_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > vol_ma[i] * 1.5
        
        if position == 0:
            if close[i] > camarilla_r3_aligned[i] and ema_slope_34_aligned[i] > 0 and vol_spike:
                # Long: bullish breakout above R3 in uptrend
                signals[i] = 0.25
                position = 1
            elif close[i] < camarilla_s3_aligned[i] and ema_slope_34_aligned[i] < 0 and vol_spike:
                # Short: bearish breakdown below S3 in downtrend
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: price returns to pivot OR EMA34 trend turns down
                if close[i] < camarilla_pivot_aligned[i] or ema_slope_34_aligned[i] <= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price returns to pivot OR EMA34 trend turns up
                if close[i] > camarilla_pivot_aligned[i] or ema_slope_34_aligned[i] >= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals