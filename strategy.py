#!/usr/bin/env python3
"""
6h Weekly Pivot + Daily Trend + Volume Spike
Hypothesis: Weekly pivot levels act as strong support/resistance in both bull and bear markets.
Breakouts above R1 or below S1 with volume confirmation and daily trend alignment yield high-probability moves.
Designed for low trade frequency (12-37/year) with clear risk control.
"""
name = "6h_WeeklyPivot_DailyTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_hlf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === WEEKLY PIVOT (R1, S1) ===
    df_1w = get_htf_data(prices, '1w')
    # Calculate weekly pivot points from weekly OHLC
    high_w = df_1w['high'].values
    low_w = df_1w['low'].values
    close_w = df_1w['close'].values
    
    # Weekly pivot: P = (H + L + C)/3
    pivot_w = (high_w + low_w + close_w) / 3.0
    # R1 = 2*P - L, S1 = 2*P - H
    r1_w = 2 * pivot_w - low_w
    s1_w = 2 * pivot_w - high_w
    
    # Align weekly levels to 6h (wait for weekly close)
    pivot_w_aligned = align_htf_to_ltf(prices, df_1w, pivot_w)
    r1_w_aligned = align_htf_to_ltf(prices, df_1w, r1_w)
    s1_w_aligned = align_htf_to_ltf(prices, df_1w, s1_w)
    
    # === DAILY TREND (EMA 34) ===
    df_1d = get_htf_data(prices, '1d')
    ema_1d = pd.Series(close_w).ewm(span=34, adjust=False, min_periods=34).mean().values  # Fix: use daily close
    # Actually get daily close from df_1d
    close_d = df_1d['close'].values
    ema_1d = pd.Series(close_d).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === 6H VOLUME (20) SPIKE ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r1_w_aligned[i]) or np.isnan(s1_w_aligned[i]) or 
            np.isnan(trend_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price > Weekly R1 + above daily EMA + volume spike
            if (close[i] > r1_w_aligned[i] and 
                close[i] > trend_1d_aligned[i] and
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price < Weekly S1 + below daily EMA + volume spike
            elif (close[i] < s1_w_aligned[i] and 
                  close[i] < trend_1d_aligned[i] and
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price crosses below weekly pivot OR volume dries up
            if close[i] < pivot_w_aligned[i] or volume[i] < vol_ma[i] * 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above weekly pivot OR volume dries up
            if close[i] > pivot_w_aligned[i] or volume[i] < vol_ma[i] * 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals