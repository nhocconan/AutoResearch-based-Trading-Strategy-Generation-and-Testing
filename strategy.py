#!/usr/bin/env python3
# 6h_Pivot_Reversion_Volume
# Hypothesis: On 6h timeframe, price reacting to daily pivot levels (S1/R1) with volume
# confirmation indicates mean-reversion opportunities. Uses daily trend filter to avoid
# trading against higher timeframe momentum. Designed for low trade frequency (~15-25/year)
# to minimize fee drag in bear markets. Works in both bull (buying dips in uptrend) and
# bear (selling rallies in downtrend) markets by aligning with daily trend.

name = "6h_Pivot_Reversion_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 1d Pivot Levels (Daily High/Low/Close) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Classic pivot: P = (H + L + C) / 3
    pivot = (high_1d + low_1d + close_1d) / 3.0
    # Support 1: S1 = 2*P - H
    s1 = 2 * pivot - high_1d
    # Resistance 1: R1 = 2*P - L
    r1 = 2 * pivot - low_1d
    
    # Align to 6h timeframe (values available after daily bar closes)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    
    # === 1d Trend Filter (Daily EMA) ===
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # === Volume Filter (20-period average) ===
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure EMA and volume MA are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(pivot_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(ema_20_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volume must be above average to ensure participation
        vol_ok = volume[i] > vol_ma[i]
        
        # Trend direction from daily EMA
        trend_up = close[i] > ema_20_1d_aligned[i]
        trend_down = close[i] < ema_20_1d_aligned[i]
        
        if position == 0:
            # LONG: Price at or below S1 with buying pressure in uptrend
            if (close[i] <= s1_aligned[i] and 
                vol_ok and 
                trend_up):
                signals[i] = 0.25
                position = 1
            # SHORT: Price at or above R1 with selling pressure in downtrend
            elif (close[i] >= r1_aligned[i] and 
                  vol_ok and 
                  trend_down):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price reaches pivot or trend changes
            if (close[i] >= pivot_aligned[i] or 
                not trend_up):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches pivot or trend changes
            if (close[i] <= pivot_aligned[i] or 
                not trend_down):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals