#!/usr/bin/env python3
# 1h_4h1d_Trend_Filtered_TripleMA_Cross
# Hypothesis: On 1h timeframe, use 4h and 1d moving averages to establish trend direction.
# Enter long when 1h EMA(8) > EMA(21) AND 4h EMA(21) > EMA(55) AND 1d EMA(55) > EMA(89).
# Enter short when 1h EMA(8) < EMA(21) AND 4h EMA(21) < EMA(55) AND 1d EMA(55) < EMA(89).
# Add volume confirmation (volume > 20-period average) and session filter (08:00-20:00 UTC).
# Exit on opposite signal or when any trend condition fails.
# Designed to reduce false signals in chop by requiring alignment across multiple timeframes.
# Works in both bull and bear markets by following the higher timeframe trend.

name = "1h_4h1d_Trend_Filtered_TripleMA_Cross"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 4h EMA (21, 55) for trend filter ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 55:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema55_4h = pd.Series(close_4h).ewm(span=55, adjust=False, min_periods=55).mean().values
    ema21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema21_4h)
    ema55_4h_aligned = align_htf_to_ltf(prices, df_4h, ema55_4h)
    
    # === 1d EMA (55, 89) for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 89:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema55_1d = pd.Series(close_1d).ewm(span=55, adjust=False, min_periods=55).mean().values
    ema89_1d = pd.Series(close_1d).ewm(span=89, adjust=False, min_periods=89).mean().values
    ema55_1d_aligned = align_htf_to_ltf(prices, df_1d, ema55_1d)
    ema89_1d_aligned = align_htf_to_ltf(prices, df_1d, ema89_1d)
    
    # === 1h EMA (8, 21) for entry signal ===
    ema8_1h = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values
    ema21_1h = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # === Volume confirmation (20-period average) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Session filter: 08:00-20:00 UTC ===
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Ensure all indicators are stable
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not (8 <= hours[i] <= 20):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Skip if any critical data is not ready
        if (np.isnan(ema8_1h[i]) or np.isnan(ema21_1h[i]) or 
            np.isnan(ema21_4h_aligned[i]) or np.isnan(ema55_4h_aligned[i]) or
            np.isnan(ema55_1d_aligned[i]) or np.isnan(ema89_1d_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend alignment conditions
        bullish_alignment = (ema8_1h[i] > ema21_1h[i]) and \
                           (ema21_4h_aligned[i] > ema55_4h_aligned[i]) and \
                           (ema55_1d_aligned[i] > ema89_1d_aligned[i])
        
        bearish_alignment = (ema8_1h[i] < ema21_1h[i]) and \
                           (ema21_4h_aligned[i] < ema55_4h_aligned[i]) and \
                           (ema55_1d_aligned[i] < ema89_1d_aligned[i])
        
        # Volume filter
        vol_ok = volume[i] > vol_ma_20[i]
        
        if position == 0:
            # LONG: bullish alignment + volume
            if bullish_alignment and vol_ok:
                signals[i] = 0.20
                position = 1
            # SHORT: bearish alignment + volume
            elif bearish_alignment and vol_ok:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # EXIT LONG: bearish alignment or volume fails
            if bearish_alignment or not vol_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: bullish alignment or volume fails
            if bullish_alignment or not vol_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals