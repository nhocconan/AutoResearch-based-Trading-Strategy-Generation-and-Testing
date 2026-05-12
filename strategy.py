#!/usr/bin/env python3
"""
1d Weekly Pivot + Daily Trend + Volume Spike
Hypothesis: Weekly pivot levels act as strong support/resistance. In trending markets (determined by daily EMA50),
price tends to respect these levels, offering high-probability bounce or breakout trades.
Volume spike confirms institutional interest. Designed for low trade frequency (<25/year) to minimize fee drag.
Works in both bull and bear markets by trading both bounces and breakouts from pivot levels.
"""
name = "1d_WeeklyPivot_DailyTrend_Volume"
timeframe = "1d"
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
    
    # === Daily EMA50 for trend filter ===
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # === Weekly Pivot Levels (from weekly OHLC) ===
    df_weekly = get_htf_data(prices, '1w')
    # Typical price = (H+L+C)/3
    pp = (df_weekly['high'] + df_weekly['low'] + df_weekly['close']) / 3
    r1 = 2 * pp - df_weekly['low']
    s1 = 2 * pp - df_weekly['high']
    r2 = pp + (df_weekly['high'] - df_weekly['low'])
    s2 = pp - (df_weekly['high'] - df_weekly['low'])
    # Align to daily timeframe (wait for weekly close)
    pp_aligned = align_htf_to_ltf(prices, df_weekly, pp.values)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1.values)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1.values)
    r2_aligned = align_htf_to_ltf(prices, df_weekly, r2.values)
    s2_aligned = align_htf_to_ltf(prices, df_weekly, s2.values)
    
    # === Volume Spike (20-day average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)  # Require 2x average volume
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure EMA50 ready
    
    for i in range(start_idx, n):
        # Skip if weekly pivot data not ready (first weeks)
        if np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG ENTRY: Price near S1/S2 + above EMA50 (uptrend) + volume spike
            near_support = (low[i] <= s1_aligned[i] * 1.002) or (low[i] <= s2_aligned[i] * 1.002)
            if near_support and close[i] > ema50[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT ENTRY: Price near R1/R2 + below EMA50 (downtrend) + volume spike
            elif (high[i] >= r1_aligned[i] * 0.998) or (high[i] >= r2_aligned[i] * 0.998):
                if close[i] < ema50[i] and vol_spike[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # EXIT LONG: Price crosses EMA50 down OR reaches R1
            if close[i] < ema50[i] or high[i] >= r1_aligned[i] * 0.998:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses EMA50 up OR reaches S1
            if close[i] > ema50[i] or low[i] <= s1_aligned[i] * 1.002:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals