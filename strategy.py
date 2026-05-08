#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy combining 1-week pivot points with 1-day volatility regime filtering.
# Weekly pivot points (calculated from prior week) provide strong institutional support/resistance.
# Long when price crosses above weekly pivot with volatility expansion (vol regime), short when crosses below.
# Uses 1-day ATR ratio to filter for volatility expansion regimes - avoids whipsaws in low volatility.
# Designed for low trade frequency (15-25/year) with clear entry/exit rules to minimize fee drag.
# Works in both bull (trend following breaks) and bear (mean reversion at pivot levels) markets.

name = "6h_WeeklyPivot_VolRegime_Breakout"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points from prior week's OHLC
    pivot = np.zeros_like(close_1w)
    r1 = np.zeros_like(close_1w)
    s1 = np.zeros_like(close_1w)
    r2 = np.zeros_like(close_1w)
    s2 = np.zeros_like(close_1w)
    r3 = np.zeros_like(close_1w)
    s3 = np.zeros_like(close_1w)
    
    for i in range(1, len(close_1w)):
        # Prior week's high, low, close
        ph = high_1w[i-1]
        pl = low_1w[i-1]
        pc = close_1w[i-1]
        
        # Standard pivot point calculation
        pivot[i] = (ph + pl + pc) / 3.0
        r1[i] = 2 * pivot[i] - pl
        s1[i] = 2 * pivot[i] - ph
        r2[i] = pivot[i] + (ph - pl)
        s2[i] = pivot[i] - (ph - pl)
        r3[i] = ph + 2 * (pivot[i] - pl)
        s3[i] = pl - 2 * (ph - pivot[i])
    
    # First week has no prior data
    pivot[0] = r1[0] = s1[0] = r2[0] = s2[0] = r3[0] = s3[0] = np.nan
    
    # Align weekly pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    # Get daily data for volatility regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1-day ATR(14) and its 30-period average for regime filter
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.maximum(np.abs(low_1d[1:] - close_1d[:-1]), tr1)
    tr = np.concatenate([[np.nan], tr2])
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_ma_30 = pd.Series(atr_14).rolling(window=30, min_periods=30).mean().values
    vol_regime = atr_14 > atr_ma_30  # Volatility expansion regime
    
    # Align volatility regime to 6h timeframe
    vol_regime_aligned = align_htf_to_ltf(prices, df_1d, vol_regime)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for alignments
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(vol_regime_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price crosses above weekly pivot with volatility expansion
            if close[i] > pivot_aligned[i] and close[i-1] <= pivot_aligned[i-1] and vol_regime_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: price crosses below weekly pivot with volatility expansion
            elif close[i] < pivot_aligned[i] and close[i-1] >= pivot_aligned[i-1] and vol_regime_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below weekly pivot OR volatility contraction
            if close[i] < pivot_aligned[i] or not vol_regime_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above weekly pivot OR volatility contraction
            if close[i] > pivot_aligned[i] or not vol_regime_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals