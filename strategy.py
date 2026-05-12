# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# 6h_Keltner_MeanReversion_RangeFilter
# Hypothesis: Mean reversion on 6h using Keltner Channel (EMA-based) with 12h trend filter.
# In ranging markets (12h ADX < 25), price tends to revert to EMA20 from Keltner extremes.
# In trending markets (12h ADX >= 25), we avoid mean reversion to prevent whipsaw.
# Works in both bull and bear by adapting to regime via ADX.
# Uses volume confirmation to avoid false reversals.

name = "6h_Keltner_MeanReversion_RangeFilter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtd_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 12h ADX for regime filter ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = np.abs(high_12h[1:] - low_12h[1:])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    dm_minus = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values (Wilder's smoothing)
    def wilder_smooth(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        result[period-1] = np.nansum(arr[1:period+1])
        for i in range(period, len(arr)):
            result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    atr = wilder_smooth(tr, 14)
    dpi = wilder_smooth(dm_plus, 14)
    dmi = wilder_smooth(dm_minus, 14)
    
    # DX and ADX
    dx = np.where((dpi + dmi) != 0, 100 * np.abs(dpi - dmi) / (dpi + dmi), np.nan)
    adx = wilder_smooth(dx, 14)
    
    # Align ADX to 6h
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # === 6h Keltner Channel (EMA20, ATRx2) ===
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    atr_6 = pd.Series(high - low).ewm(span=6, adjust=False, min_periods=6).mean().values
    upper_keltner = ema20 + 2 * atr_6
    lower_keltner = ema20 - 2 * atr_6
    
    # === Volume confirmation (20-period average) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema20[i]) or np.isnan(upper_keltner[i]) or np.isnan(lower_keltner[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Range regime: 12h ADX < 25
        range_market = adx_aligned[i] < 25
        
        # Volume filter: above average
        vol_ok = volume[i] > vol_ma_20[i]
        
        if position == 0:
            # LONG: price at lower Keltner, in range, volume confirmation
            if range_market and vol_ok and close[i] <= lower_keltner[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: price at upper Keltner, in range, volume confirmation
            elif range_market and vol_ok and close[i] >= upper_keltner[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: price crosses EMA20 or regime changes to trend
            if close[i] >= ema20[i] or not range_market:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price crosses EMA20 or regime changes to trend
            if close[i] <= ema20[i] or not range_market:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals