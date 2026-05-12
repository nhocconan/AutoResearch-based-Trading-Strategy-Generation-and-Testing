#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_12hTrend_Volume
# Hypothesis: Camarilla pivot R1/S1 breakouts on 4h with 12h EMA trend filter and volume confirmation.
# Works in bull (breakouts follow trend) and bear (mean-reversion at extremes via trend filter).
# Target: 20-40 trades/year. Uses proven Camarilla structure with volume and trend filters.

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_Volume"
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
    
    # === 12h Trend Filter ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # 50-period EMA on 12h for trend direction
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # === Camarilla Pivot Levels (from previous day) ===
    # Calculate daily pivot from previous day's OHLC
    # We'll use the daily timeframe for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_day_high = df_1d['high'].shift(1).values  # previous day high
    prev_day_low = df_1d['low'].shift(1).values    # previous day low
    prev_day_close = df_1d['close'].shift(1).values # previous day close
    
    # Align daily data to 4h timeframe
    prev_day_high_aligned = align_htf_to_ltf(prices, df_1d, prev_day_high)
    prev_day_low_aligned = align_htf_to_ltf(prices, df_1d, prev_day_low)
    prev_day_close_aligned = align_htf_to_ltf(prices, df_1d, prev_day_close)
    
    # Camarilla calculations
    # Pivot = (H + L + C) / 3
    pivot = (prev_day_high_aligned + prev_day_low_aligned + prev_day_close_aligned) / 3.0
    # Range = H - L
    range_val = prev_day_high_aligned - prev_day_low_aligned
    
    # R1 = C + (H-L) * 1.1/12
    r1 = prev_day_close_aligned + range_val * 1.1 / 12.0
    # S1 = C - (H-L) * 1.1/12
    s1 = prev_day_close_aligned - range_val * 1.1 / 12.0
    
    # === Volume Confirmation (20-period average) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(r1[i]) or np.isnan(s1[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(prev_day_high_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend direction
        trend_up = close[i] > ema_50_12h_aligned[i]
        trend_down = close[i] < ema_50_12h_aligned[i]
        
        # Volume filter: above average
        vol_ok = volume[i] > vol_ma_20[i]
        
        if position == 0:
            # LONG: Price breaks above R1 with volume and uptrend
            if (close[i] > r1[i] and vol_ok and trend_up):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 with volume and downtrend
            elif (close[i] < s1[i] and vol_ok and trend_down):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price returns to pivot or trend changes
            if (close[i] < pivot[i] or not trend_up):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to pivot or trend changes
            if (close[i] > pivot[i] or not trend_down):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals