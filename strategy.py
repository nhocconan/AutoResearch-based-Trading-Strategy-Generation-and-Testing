#!/usr/bin/env python3
# 6h_1d_Pivot_R4S4_Breakout_Volume_TrendFilter
# Hypothesis: Breakout beyond daily pivot R4/S4 with volume confirmation and 1w trend filter.
# R4/S4 represent strong breakout levels; trading in direction of 1w trend avoids counter-trend whipsaws.
# Works in bull/bear via 1w trend filter - only trade with the weekly trend.
# Target: 60-120 trades over 4 years (15-30/year).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_Pivot_R4S4_Breakout_Volume_TrendFilter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop for pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # === Calculate 1d pivot levels (R4, S4) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point and range
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels: R4 = close + (range * 1.1/2), S4 = close - (range * 1.1/2)
    r4_1d = close_1d + (range_1d * 1.1 / 2)
    s4_1d = close_1d - (range_1d * 1.1 / 2)
    
    # === 1w trend filter: EMA(34) direction ===
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_up_1w = ema_34_1w > np.roll(ema_34_1w, 1)
    trend_up_1w[0] = False  # First value has no previous
    
    # === 6h: Volume ratio (current vs 20-period average) ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # Align all 1d levels and 1w trend to 6h
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    trend_up_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_up_1w.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):  # Start after EMA warmup
        # Get values
        close_val = prices['close'].iloc[i]
        r4_1d_val = r4_1d_aligned[i]
        s4_1d_val = s4_1d_aligned[i]
        vol_ratio_val = vol_ratio[i]
        trend_up_1w_val = trend_up_1w_aligned[i] > 0.5
        
        # Skip if any value is NaN
        if (np.isnan(r4_1d_val) or np.isnan(s4_1d_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above R4 with volume confirmation and 1w uptrend
            if (close_val > r4_1d_val and  # Price broke above R4
                vol_ratio_val > 2.0 and  # Volume confirmation
                trend_up_1w_val):  # 1w uptrend filter
                signals[i] = 0.25
                position = 1
            # Short: Break below S4 with volume confirmation and 1w downtrend
            elif (close_val < s4_1d_val and  # Price broke below S4
                  vol_ratio_val > 2.0 and  # Volume confirmation
                  not trend_up_1w_val):  # 1w downtrend filter
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns below R4 or shows weakness
            if close_val < r4_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns above S4 or shows weakness
            if close_val > s4_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals