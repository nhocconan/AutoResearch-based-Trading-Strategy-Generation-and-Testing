#!/usr/bin/env python3
"""
6h Weekly Pivot + 1d Trend Filter + Volume Spike
Hypothesis: Weekly pivot levels act as strong support/resistance zones. When price breaks
above/below weekly pivot with alignment to daily trend (EMA34) and volume confirmation,
it signals continuation of the trend. This strategy targets low-frequency, high-conviction
breakouts suitable for 6h timeframe, aiming for 15-30 trades/year to minimize fee drag.
Works in both bull and bear markets by following the trend defined by daily EMA.
"""
name = "6h_WeeklyPivot_1dTrend_Volume"
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
    
    # === Weekly Pivot Calculation (from weekly data) ===
    df_1w = get_htf_data(prices, '1w')  # Call ONCE before loop
    # Typical price for weekly pivot: (H + L + C) / 3
    typical_price = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3.0
    weekly_pivot = typical_price.values
    # Support and resistance levels
    weekly_r1 = 2 * weekly_pivot - df_1w['low'].values
    weekly_s1 = 2 * weekly_pivot - df_1w['high'].values
    weekly_r2 = weekly_pivot + (df_1w['high'].values - df_1w['low'].values)
    weekly_s2 = weekly_pivot - (df_1w['high'].values - df_1w['low'].values)
    # Align to 6h timeframe (waits for weekly close)
    pivot_6h = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    r1_6h = align_htf_to_ltf(prices, df_1w, weekly_r1)
    s1_6h = align_htf_to_ltf(prices, df_1w, weekly_s1)
    r2_6h = align_htf_to_ltf(prices, df_1w, weekly_r2)
    s2_6h = align_htf_to_ltf(prices, df_1w, weekly_s2)
    
    # === Daily EMA34 for Trend Filter ===
    df_1d = get_htf_data(prices, '1d')  # Call ONCE before loop
    ema_34 = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_6h = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # === Volume Spike (20-period) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)  # Require 2x average volume for confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(pivot_6h[i]) or np.isnan(r1_6h[i]) or np.isnan(s1_6h[i]) or
            np.isnan(r2_6h[i]) or np.isnan(s2_6h[i]) or np.isnan(ema_34_6h[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above weekly R1, above daily EMA34, with volume spike
            if (close[i] > r1_6h[i] and 
                close[i] > ema_34_6h[i] and
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below weekly S1, below daily EMA34, with volume spike
            elif (close[i] < s1_6h[i] and 
                  close[i] < ema_34_6h[i] and
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price breaks below weekly pivot OR volume dries up
            if close[i] < pivot_6h[i] or not vol_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above weekly pivot OR volume dries up
            if close[i] > pivot_6h[i] or not vol_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals