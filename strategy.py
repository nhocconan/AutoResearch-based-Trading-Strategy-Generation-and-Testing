#!/usr/bin/env python3
# 12h_camarilla_pivot_volume_chop_v1
# Hypothesis: 12h strategy using 1d Camarilla pivot levels for support/resistance,
# with volume confirmation (>1.5x 20-period average) and choppiness regime filter.
# Long when price touches S3 level with volume confirmation in trending regime (CHOP < 38.2).
# Short when price touches R3 level with volume confirmation in trending regime (CHOP < 38.2).
# Uses discrete sizing (±0.25) to minimize fee churn. Target: 12-37 trades/year.
# Uses 1d HTF data for Camarilla levels and choppiness, called ONCE before loop.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for Camarilla pivot and choppiness
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need enough for choppiness calculation
        return np.zeros(n)
    
    high_d = df_1d['high'].values
    low_d = df_1d['low'].values
    close_d = df_1d['close'].values
    
    # Daily Camarilla pivot levels (based on previous day)
    # Pivot = (high + low + close) / 3
    pivot = (high_d + low_d + close_d) / 3.0
    # Range = high - low
    rng = high_d - low_d
    
    # Camarilla levels
    # S3 = close - (range * 1.1/4)
    s3 = close_d - (rng * 1.1 / 4.0)
    # S2 = close - (range * 1.1/6)
    s2 = close_d - (rng * 1.1 / 6.0)
    # S1 = close - (range * 1.1/12)
    s1 = close_d - (rng * 1.1 / 12.0)
    # R1 = close + (range * 1.1/12)
    r1 = close_d + (rng * 1.1 / 12.0)
    # R2 = close + (range * 1.1/6)
    r2 = close_d + (rng * 1.1 / 6.0)
    # R3 = close + (range * 1.1/4)
    r3 = close_d + (rng * 1.1 / 4.0)
    
    # Daily Choppiness Index (CHOP) - measures trend vs range
    # CHOP = 100 * log10(sum(ATR) / (max(high) - min(low))) / log10(n)
    # Using 14-period as standard
    tr1 = high_d[1:] - low_d[1:]
    tr2 = np.abs(high_d[1:] - close_d[:-1])
    tr3 = np.abs(low_d[1:] - close_d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align with index 0
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_atr = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high_d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_atr / (max_high - min_low)) / np.log10(14)
    
    # Align daily data to 12h timeframe
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # 12h volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(s3_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r1_aligned[i]) or np.isnan(r2_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(pivot_aligned[i]) or np.isnan(chop_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        # Trending regime: CHOP < 38.2 (strong trend)
        trending_regime = chop_aligned[i] < 38.2
        
        if position == 1:  # Long position
            # Exit: price moves above S2 (profit target) OR below S1 (stop loss)
            if close[i] > s2_aligned[i] or close[i] < s1_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price moves below R2 (profit target) OR above R1 (stop loss)
            if close[i] < r2_aligned[i] or close[i] > r1_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if volume_confirmed and trending_regime:
                # Long entry: price touches or crosses below S3 with volume confirmation
                if low[i] <= s3_aligned[i] and close[i] > s3_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price touches or crosses above R3 with volume confirmation
                elif high[i] >= r3_aligned[i] and close[i] < r3_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals