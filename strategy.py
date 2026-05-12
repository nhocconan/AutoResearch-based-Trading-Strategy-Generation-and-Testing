#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_1dTrend_Volume
# Hypothesis: Camarilla pivot level breakouts on 4h with 1d EMA trend filter and volume confirmation.
# Uses Camarilla R1/S1 levels for precise entry/exit with EMA trend filter and volume confirmation.
# Works in bull (breakouts continue) and bear (mean-reversion via trend filter).
# Target: 20-40 trades/year.

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "4h"
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
    
    # === 1d Trend Filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 34-period EMA on 1d for trend direction
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === Camarilla Pivot Levels from 1d (calculated on daily data) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_vals = df_1d['close'].values
    
    # Calculate pivot point and Camarilla levels
    pivot = (high_1d + low_1d + close_1d_vals) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels: R1 = close + (range * 1.1/12), S1 = close - (range * 1.1/12)
    r1 = close_1d_vals + (range_1d * 1.1 / 12)
    s1 = close_1d_vals - (range_1d * 1.1 / 12)
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === Volume Confirmation (20-period average) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend direction
        trend_up = close[i] > ema_34_1d_aligned[i]
        trend_down = close[i] < ema_34_1d_aligned[i]
        
        # Volume filter: above average
        vol_ok = volume[i] > vol_ma_20[i]
        
        if position == 0:
            # LONG: Price breaks above R1 with volume and uptrend
            if (close[i] > r1_aligned[i] and vol_ok and trend_up):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 with volume and downtrend
            elif (close[i] < s1_aligned[i] and vol_ok and trend_down):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price returns below S1 or trend changes
            if (close[i] < s1_aligned[i] or not trend_up):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns above R1 or trend changes
            if (close[i] > r1_aligned[i] or not trend_down):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals