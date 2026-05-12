#!/usr/bin/env python3
name = "6h_Donchian20_WeeklyPivotTrend_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1D DATA FOR DAILY TREND FILTER ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # === 7D DATA FOR WEEKLY PIVOT ===
    df_7d = get_htf_data(prices, '7d')
    high_7d = df_7d['high'].values
    low_7d = df_7d['low'].values
    close_7d = df_7d['close'].values
    
    # === DAILY TREND: EMA50 (strong trend filter) ===
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_6h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # === WEEKLY PIVOT POINTS (using prior week's OHLC) ===
    # Standard pivot: PP = (H + L + C)/3
    # R1 = 2*PP - L, S1 = 2*PP - H
    # R2 = PP + (H - L), S2 = PP - (H - L)
    # R3 = H + 2*(PP - L), S3 = L - 2*(H - PP)
    pivot_7d = (high_7d + low_7d + close_7d) / 3.0
    r1_7d = 2 * pivot_7d - low_7d
    s1_7d = 2 * pivot_7d - high_7d
    r2_7d = pivot_7d + (high_7d - low_7d)
    s2_7d = pivot_7d - (high_7d - low_7d)
    r3_7d = high_7d + 2 * (pivot_7d - low_7d)
    s3_7d = low_7d - 2 * (high_7d - pivot_7d)
    
    # Align weekly pivots to 6h timeframe (wait for weekly close)
    pivot_6h = align_htf_to_ltf(prices, df_7d, pivot_7d)
    r1_6h = align_htf_to_ltf(prices, df_7d, r1_7d)
    s1_6h = align_htf_to_ltf(prices, df_7d, s1_7d)
    r2_6h = align_htf_to_ltf(prices, df_7d, r2_7d)
    s2_6h = align_htf_to_ltf(prices, df_7d, s2_7d)
    r3_6h = align_htf_to_ltf(prices, df_7d, r3_7d)
    s3_6h = align_htf_to_ltf(prices, df_7d, s3_7d)
    
    # === 6H DONCHIAN CHANNEL (20-period) ===
    # Highest high and lowest low over 20 periods
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_20 + lowest_20) / 2.0
    
    # === VOLUME CONFIRMATION (20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.8)  # Strong volume confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 50)  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema50_1d_6h[i]) or np.isnan(pivot_6h[i]) or np.isnan(r1_6h[i]) or 
            np.isnan(s1_6h[i]) or np.isnan(r2_6h[i]) or np.isnan(s2_6h[i]) or
            np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or np.isnan(highest_20[i]) or
            np.isnan(lowest_20[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian upper + above weekly R1 + above daily EMA50 + volume
            if (close[i] > highest_20[i] and 
                close[i] > r1_6h[i] and
                close[i] > ema50_1d_6h[i] and
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian lower + below weekly S1 + below daily EMA50 + volume
            elif (close[i] < lowest_20[i] and 
                  close[i] < s1_6h[i] and
                  close[i] < ema50_1d_6h[i] and
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price falls below Donchian middle OR below weekly S1
            if close[i] < donchian_mid[i] or close[i] < s1_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price rises above Donchian middle OR above weekly R1
            if close[i] > donchian_mid[i] or close[i] > r1_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals