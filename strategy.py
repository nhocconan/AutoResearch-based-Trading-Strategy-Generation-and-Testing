#!/usr/bin/env python3
# 6h Weekly Pivot Breakout with Volume and 1d EMA Trend Filter
# Hypothesis: Price breaking above weekly R4 or below weekly S4 with volume
# confirmation indicates strong momentum, filtered by 1d EMA50 trend.
# Weekly pivots capture institutional levels; breakouts avoid false signals.
# Works in bull/bear by following breakout direction with trend alignment.
# Target: 12-37 trades/year (50-150 over 4 years) with low fee drag.

name = "6h_WeeklyPivot_Breakout_Volume_Trend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Weekly Data for Pivot Points ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Calculate weekly pivot points (standard formula)
    pp = (weekly_high + weekly_low + weekly_close) / 3.0
    r1 = 2 * pp - weekly_low
    s1 = 2 * pp - weekly_high
    r2 = pp + (weekly_high - weekly_low)
    s2 = pp - (weekly_high - weekly_low)
    r3 = weekly_high + 2 * (pp - weekly_low)
    s3 = weekly_low - 2 * (weekly_high - pp)
    r4 = weekly_high + 3 * (weekly_high - weekly_low)
    s4 = weekly_low - 3 * (weekly_high - weekly_low)
    
    # Align weekly pivot levels to 6h timeframe (wait for weekly close)
    pp_6h = align_htf_to_ltf(prices, df_1w, pp)
    r4_6h = align_htf_to_ltf(prices, df_1w, r4)
    s4_6h = align_htf_to_ltf(prices, df_1w, s4)
    
    # === Daily Data for EMA Trend Filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    daily_close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(daily_close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_6h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === Volume Spike (20-period on 6h) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(pp_6h[i]) or np.isnan(r4_6h[i]) or np.isnan(s4_6h[i]) or 
            np.isnan(ema_50_6h[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Break above weekly R4 + volume spike + price above daily EMA50
            if (close[i] > r4_6h[i] and 
                vol_spike[i] and
                close[i] > ema_50_6h[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below weekly S4 + volume spike + price below daily EMA50
            elif (close[i] < s4_6h[i] and 
                  vol_spike[i] and
                  close[i] < ema_50_6h[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price crosses below weekly pivot point
            if close[i] < pp_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above weekly pivot point
            if close[i] > pp_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals