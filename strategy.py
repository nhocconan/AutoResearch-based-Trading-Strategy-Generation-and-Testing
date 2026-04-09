#!/usr/bin/env python3
# 12h_camarilla_1d_volume_chop_v1
# Hypothesis: 12h strategy using daily Camarilla pivot levels with volume confirmation
# and 1-week choppiness regime filter. Enters long when price touches S3 level with
# volume spike in choppy market (CHOP > 61.8), short when touches R3 level with
# volume spike. Uses discrete sizing (±0.25) to minimize fee churn. Designed for
# low trade frequency (target: 50-150 total trades over 4 years) to avoid fee drag
# and work in both bull/bear markets via regime adaptation.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_1d_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily Camarilla pivot levels (based on previous day)
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # S3 = C - (Range * 1.1000 / 2)
    # S2 = C - (Range * 1.1000 / 4)
    # S1 = C - (Range * 1.1000 / 6)
    # R1 = C + (Range * 1.1000 / 6)
    # R2 = C + (Range * 1.1000 / 4)
    # R3 = C + (Range * 1.1000 / 2)
    
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    s3_1d = close_1d - (range_1d * 1.1000 / 2.0)
    s2_1d = close_1d - (range_1d * 1.1000 / 4.0)
    s1_1d = close_1d - (range_1d * 1.1000 / 6.0)
    r1_1d = close_1d + (range_1d * 1.1000 / 6.0)
    r2_1d = close_1d + (range_1d * 1.1000 / 4.0)
    r3_1d = close_1d + (range_1d * 1.1000 / 2.0)
    
    # Align Camarilla levels to 12h timeframe (use previous day's levels)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    
    # 1w HTF data for choppiness regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Choppiness Index (14-period)
    # True Range
    tr1 = pd.Series(high_1w).shift(1) - pd.Series(low_1w).shift(1)
    tr2 = abs(pd.Series(high_1w) - pd.Series(close_1w).shift(1))
    tr3 = abs(pd.Series(low_1w) - pd.Series(close_1w).shift(1))
    tr_1w = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # ATR = smoothed TR
    atr_1w = pd.Series(tr_1w).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Highest high and lowest low over 14 periods
    hh_1w = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    ll_1w = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(sum(ATR14) / (HH - LL)) / log10(14)
    sum_atr_1w = pd.Series(atr_1w).rolling(window=14, min_periods=14).sum().values
    chop_1w = 100 * (np.log10(sum_atr_1w) - np.log10(hh_1w - ll_1w + 1e-10)) / np.log10(14)
    chop_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any required data is NaN
        if (np.isnan(s3_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or
            np.isnan(chop_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0x 20-period average
        if i >= 20:
            volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().iloc[i]
            volume_confirmed = volume[i] > 2.0 * volume_ma if not np.isnan(volume_ma) else False
        else:
            volume_confirmed = False
        
        # Regime filter: choppy market (CHOP > 61.8 = ranging)
        ranging = chop_1w_aligned[i] > 61.8
        
        if position == 1:  # Long position
            # Exit: price moves above S2 (profit target) or below S3 (stop)
            if close[i] > s2_1d_aligned[i] or close[i] < s3_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price moves below R2 (profit target) or above R3 (stop)
            if close[i] < r2_1d_aligned[i] or close[i] > r3_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if volume_confirmed and ranging:
                # Long conditions: price touches or crosses below S3 with volume
                if close[i] <= s3_1d_aligned[i] and low[i] <= s3_1d_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short conditions: price touches or crosses above R3 with volume
                elif close[i] >= r3_1d_aligned[i] and high[i] >= r3_1d_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals