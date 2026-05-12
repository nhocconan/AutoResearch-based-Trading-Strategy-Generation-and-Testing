#!/usr/bin/env python3
"""
12h Daily Camarilla Pivot + Volume Spike + Chop Filter
Hypothesis: Daily Camarilla pivot levels (from daily chart) act as strong support/resistance.
In trending markets, price respects these levels as pullback entries.
In ranging markets, reversals occur at these levels.
Volume confirmation filters false breakouts.
Choppiness filter ensures we only trade in trending or clear ranging conditions.
Timeframe: 12h balances trade frequency (~12-37/year) with signal quality.
Works in bull/bear: uses price action at key levels rather than trend direction.
"""

name = "12h_DailyCamarilla_Pivot_Volume_Chop"
timeframe = "12h"
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
    
    # === DAILY DATA FOR CAMARILLA PIVOT LEVELS ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla pivot points calculation
    # P = (H + L + C) / 3
    # R4 = C + ((H-L) * 1.1/2)
    # R3 = C + ((H-L) * 1.1/4)
    # R2 = C + ((H-L) * 1.1/6)
    # R1 = C + ((H-L) * 1.1/12)
    # S1 = C - ((H-L) * 1.1/12)
    # S2 = C - ((H-L) * 1.1/6)
    # S3 = C - ((H-L) * 1.1/4)
    # S4 = C - ((H-L) * 1.1/2)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    r4_1d = close_1d + (range_1d * 1.1 / 2)
    r3_1d = close_1d + (range_1d * 1.1 / 4)
    r2_1d = close_1d + (range_1d * 1.1 / 6)
    r1_1d = close_1d + (range_1d * 1.1 / 12)
    s1_1d = close_1d - (range_1d * 1.1 / 12)
    s2_1d = close_1d - (range_1d * 1.1 / 6)
    s3_1d = close_1d - (range_1d * 1.1 / 4)
    s4_1d = close_1d - (range_1d * 1.1 / 2)
    
    # Align daily levels to 12h timeframe
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # === VOLUME CONFIRMATION (24-period for 12h) ===
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (vol_ma * 2.0)  # Strong volume filter to reduce trades
    
    # === CHOPPINESS FILTER (14-period) ===
    # Calculate True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0  # First value has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR (14)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Chop = 100 * log10(sum(TR,14) / (max(high,14) - min(low,14))) / log10(14)
    def rolling_max(arr, window):
        return pd.Series(arr).rolling(window=window, min_periods=1).max().values
    
    def rolling_min(arr, window):
        return pd.Series(arr).rolling(window=window, min_periods=1).min().values
    
    sum_tr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    max_high = rolling_max(high, 14)
    min_low = rolling_min(low, 14)
    range_14 = max_high - min_low
    
    # Avoid division by zero
    chop = np.full_like(close, 50.0, dtype=float)
    mask = range_14 > 0
    chop[mask] = 100 * np.log10(sum_tr[mask] / range_14[mask]) / np.log10(14)
    
    # Chop thresholds: >61.8 = ranging, <38.2 = trending
    chop_ranging = chop > 61.8
    chop_trending = chop < 38.2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 24  # For volume MA and chop
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(pivot_1d_aligned[i]) or np.isnan(r1_1d_aligned[i]) or 
            np.isnan(r2_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or 
            np.isnan(r4_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or 
            np.isnan(s2_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or 
            np.isnan(s4_1d_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price bounces off S1 or S2 with volume spike
            long_condition = (((close[i] > s1_1d_aligned[i] and low[i] <= s1_1d_aligned[i] * 1.002) or
                              (close[i] > s2_1d_aligned[i] and low[i] <= s2_1d_aligned[i] * 1.002)) and
                             volume_spike[i] and
                             (chop_ranging[i] or chop_trending[i]))  # Trade in both regimes
            
            # SHORT: Price rejects at R1 or R2 with volume spike
            short_condition = (((close[i] < r1_1d_aligned[i] and high[i] >= r1_1d_aligned[i] * 0.998) or
                               (close[i] < r2_1d_aligned[i] and high[i] >= r2_1d_aligned[i] * 0.998)) and
                              volume_spike[i] and
                              (chop_ranging[i] or chop_trending[i]))
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price breaks below S2 or reaches R2
            if close[i] < s2_1d_aligned[i] or close[i] > r2_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R2 or reaches S2
            if close[i] > r2_1d_aligned[i] or close[i] < s2_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals