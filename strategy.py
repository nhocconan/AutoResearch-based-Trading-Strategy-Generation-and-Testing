#!/usr/bin/env python3
"""
6h_GoldenCross_WeeklyTrend_Retrace
Hypothesis: 6h EMA crossover (golden/death cross) with weekly trend filter and volume confirmation.
- Golden cross: EMA20 > EMA50 on 6h
- Death cross: EMA20 < EMA50 on 6h
- Weekly trend: EMA50 on weekly timeframe (trend filter: price > weekly EMA50 for longs, < for shorts)
- Volume filter: current volume > 1.5 * 20-period average on 6h
- Entry: Golden cross + weekly uptrend + volume spike (long), Death cross + weekly downtrend + volume spike (short)
- Exit: Opposite cross or trend failure
- Designed to work in bull/bear by using weekly trend filter to avoid counter-trend trades
- Target: 20-40 trades/year on 6h (80-160 total over 4 years)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 6h EMA20 and EMA50 for golden/death cross
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Weekly trend filter (EMA50)
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    close_weekly = df_weekly['close'].values
    ema50_weekly = pd.Series(close_weekly).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema50_weekly)
    
    # Volume spike: current volume > 1.5 * 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough data for calculations
    start_idx = max(50, 20) + 1
    
    for i in range(start_idx, n):
        if (np.isnan(ema20[i]) or np.isnan(ema50[i]) or np.isnan(ema50_weekly_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: golden cross + weekly uptrend + volume spike
            if (ema20[i] > ema50[i] and close[i] > ema50_weekly_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: death cross + weekly downtrend + volume spike
            elif (ema20[i] < ema50[i] and close[i] < ema50_weekly_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: death cross or weekly trend failure
            if (ema20[i] < ema50[i] or close[i] < ema50_weekly_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: golden cross or weekly trend failure
            if (ema20[i] > ema50[i] or close[i] > ema50_weekly_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_GoldenCross_WeeklyTrend_Retrace"
timeframe = "6h"
leverage = 1.0