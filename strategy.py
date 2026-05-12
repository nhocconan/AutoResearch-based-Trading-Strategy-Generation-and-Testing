#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_1dTrend_Volume
# Hypothesis: Camarilla R1/S1 breakout on 4h with 1d EMA trend filter and volume confirmation.
# Camarilla levels provide statistically significant support/resistance from prior day's range.
# The 1d EMA filter ensures alignment with daily trend, reducing counter-trend trades.
# Volume confirmation ensures breakouts have conviction. Designed to work in both bull and bear markets
# by following the trend defined by higher timeframe.

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
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 34-period EMA on 1d for trend direction
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === Camarilla Levels (based on prior day's range) ===
    # Calculate daily high/low/close for Camarilla calculation
    # We'll use the 1d data to compute Camarilla levels for the current 4h bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_vals = df_1d['close'].values
    
    # Camarilla R1, S1, R2, S2, R3, S3, R4, S4
    # R4 = Close + 1.5 * (High - Low)
    # R3 = Close + 1.25 * (High - Low)
    # R2 = Close + 1.166 * (High - Low)
    # R1 = Close + 1.083 * (High - Low)
    # S1 = Close - 1.083 * (High - Low)
    # S2 = Close - 1.166 * (High - Low)
    # S3 = Close - 1.25 * (High - Low)
    # S4 = Close - 1.5 * (High - Low)
    
    camarilla_r1 = close_1d_vals + 1.083 * (high_1d - low_1d)
    camarilla_s1 = close_1d_vals - 1.083 * (high_1d - low_1d)
    camarilla_r2 = close_1d_vals + 1.166 * (high_1d - low_1d)
    camarilla_s2 = close_1d_vals - 1.166 * (high_1d - low_1d)
    camarilla_r3 = close_1d_vals + 1.25 * (high_1d - low_1d)
    camarilla_s3 = close_1d_vals - 1.25 * (high_1d - low_1d)
    camarilla_r4 = close_1d_vals + 1.5 * (high_1d - low_1d)
    camarilla_s4 = close_1d_vals - 1.5 * (high_1d - low_1d)
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
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
        
        # Trend direction from 1d EMA
        trend_up = close[i] > ema_34_1d_aligned[i]
        trend_down = close[i] < ema_34_1d_aligned[i]
        
        # Volume filter: above average
        vol_ok = volume[i] > vol_ma_20[i]
        
        if position == 0:
            # LONG: Price breaks above R1 with volume and daily uptrend
            if (close[i] > r1_aligned[i] and vol_ok and trend_up):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 with volume and daily downtrend
            elif (close[i] < s1_aligned[i] and vol_ok and trend_down):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price closes below S1 or daily trend changes
            if (close[i] < s1_aligned[i] or not trend_up):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above R1 or daily trend changes
            if (close[i] > r1_aligned[i] or not trend_down):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals