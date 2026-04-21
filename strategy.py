#!/usr/bin/env python3
"""
12h_1d_1w_Camarilla_R1S1_Breakout_Volume_Regime_v1
Hypothesis: Breakout above Camarilla R1 (bullish) or below S1 (bearish) on 12h timeframe,
confirmed by 1d volume spike and 1w choppiness regime (CHOP < 38.2 = trending).
Uses discrete position sizing (0.25) to limit fee drag. Designed for low trade frequency
(12-37/year) to work in both bull and bear markets by following 1d trend via 1w regime filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data once for Camarilla levels (from previous day)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Camarilla levels: R1, S1, R2, S2, R3, S3, R4, S4
    rang = prev_high - prev_low
    r1 = prev_close + rang * 1.0 / 12
    s1 = prev_close - rang * 1.0 / 12
    r2 = prev_close + rang * 2.0 / 12
    s2 = prev_close - rang * 2.0 / 12
    r3 = prev_close + rang * 3.0 / 12
    s3 = prev_close - rang * 3.0 / 12
    r4 = prev_close + rang * 4.0 / 12
    s4 = prev_close - rang * 4.0 / 12
    
    # Align Camarilla levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Load 1d data for volume confirmation
    vol_1d = df_1d['volume'].values
    # 20-period average volume on 1d
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Load 1w data for choppiness regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range for choppiness
    tr1 = np.abs(high_1w - low_1w)
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    
    # Sum of True Range over 14 periods
    tr_sum_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: CHOP = 100 * log10(tr_sum_14 / (hh_14 - ll_14)) / log10(14)
    # Avoid division by zero
    hl_range_14 = hh_14 - ll_14
    chop = np.where(hl_range_14 > 0, 100 * np.log10(tr_sum_14 / hl_range_14) / np.log10(14), 50)
    chop = chop.astype(float)
    
    # Align choppiness to 12h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(vol_ma_aligned[i]) or np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation: current volume > 1.5 * 20-day average
        volume_ok = volume > 1.5 * vol_ma_aligned[i]
        
        # Regime filter: only trade in trending markets (CHOP < 38.2)
        regime_ok = chop_aligned[i] < 38.2
        
        if position == 0:
            # Long: break above R1 with volume and regime confirmation
            if price > r1_aligned[i] and volume_ok and regime_ok:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with volume and regime confirmation
            elif price < s1_aligned[i] and volume_ok and regime_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: reverse below R1 or reach R2 (take profit)
            if price < r1_aligned[i] or price > r2_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: reverse above S1 or reach S2 (take profit)
            if price > s1_aligned[i] or price < s2_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1d_1w_Camarilla_R1S1_Breakout_Volume_Regime_v1"
timeframe = "12h"
leverage = 1.0