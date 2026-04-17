#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R1/S1 breakout with volume confirmation and 1d chop regime filter.
Long when price breaks above Camarilla R1 AND volume > 1.3x average AND CHOP > 61.8 (ranging).
Short when price breaks below Camarilla S1 AND volume > 1.3x average AND CHOP > 61.8.
Exit when price reverts to Camarilla midpoint (PP) OR CHOP < 38.2 (trending).
Uses 4h for Camarilla calculation and 1d for CHOP filter to avoid whipsaw in strong trends.
Target: 75-200 total trades over 4 years (19-50/year). Camarilla levels provide precise intraday
support/resistance, volume confirmation filters breakout validity, CHOP filter ensures mean-reversion
edge works best in ranging markets while avoiding strong trending periods where breakouts fail.
"""

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
    
    # Get 4h data for Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla levels on 4h timeframe (based on previous day)
    # Camarilla uses previous period's high, low, close
    prev_high = np.roll(high_4h, 1)
    prev_low = np.roll(low_4h, 1)
    prev_close = np.roll(close_4h, 1)
    # First value will be NaN due to roll, handled by min_periods later
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    r1 = pivot + (range_hl * 1.1 / 12)
    s1 = pivot - (range_hl * 1.1 / 12)
    r2 = pivot + (range_hl * 1.1 / 6)
    s2 = pivot - (range_hl * 1.1 / 6)
    r3 = pivot + (range_hl * 1.1 / 4)
    s3 = pivot - (range_hl * 1.1 / 4)
    r4 = pivot + (range_hl * 1.1 / 2)
    s4 = pivot - (range_hl * 1.1 / 2)
    pp = pivot  # Camarilla midpoint
    
    # Get 1d data for CHOP filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate CHOP (Choppiness Index) on 1d timeframe (14-period)
    high_1d_series = pd.Series(high_1d)
    low_1d_series = pd.Series(low_1d)
    close_1d_series = pd.Series(close_1d)
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    # ATR (14-period)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = high_1d_series.rolling(window=14, min_periods=14).max().values
    ll_14 = low_1d_series.rolling(window=14, min_periods=14).min().values
    
    # CHOP formula: 100 * log10(sum(ATR14) / (HH14 - LL14)) / log10(14)
    # Avoid division by zero
    range_14 = hh_14 - ll_14
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    
    chop = np.zeros_like(close_1d)
    mask = (range_14 > 0) & (~np.isnan(range_14)) & (~np.isnan(sum_atr_14))
    chop[mask] = 100 * np.log10(sum_atr_14[mask] / range_14[mask]) / np.log10(14)
    # For invalid cases, set to 50 (neutral)
    chop[~mask] = 50.0
    
    # Align 4h Camarilla to 4h timeframe (no alignment needed)
    r1_aligned = r1
    s1_aligned = s1
    pp_aligned = pp
    
    # Align 1d CHOP to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume average (20-period) on 4h
    volume_4h = df_4h['volume'].values
    volume_ma = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_4h, volume_ma)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(pp_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        pp_val = pp_aligned[i]
        chop_val = chop_aligned[i]
        vol_ma = volume_ma_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: price > Camarilla R1 AND volume > 1.3x avg AND CHOP > 61.8 (ranging)
            if price > r1_val and vol > 1.3 * vol_ma and chop_val > 61.8:
                signals[i] = 0.25
                position = 1
            # Short: price < Camarilla S1 AND volume > 1.3x avg AND CHOP > 61.8 (ranging)
            elif price < s1_val and vol > 1.3 * vol_ma and chop_val > 61.8:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price < Camarilla PP OR CHOP < 38.2 (trending)
            if price < pp_val or chop_val < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price > Camarilla PP OR CHOP < 38.2 (trending)
            if price > pp_val or chop_val < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Volume_CHOP_Filter"
timeframe = "4h"
leverage = 1.0