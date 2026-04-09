#!/usr/bin/env python3
# 1d_camarilla_pivot_volume_chop_v1
# Hypothesis: Daily strategy using Camarilla pivot levels from 1d with volume confirmation and weekly chop regime filter.
# Enters long when price breaks above H3 level with volume spike, short when breaks below L3 level with volume spike.
# Uses weekly choppiness index to avoid ranging markets (CHOP > 61.8 = range, no trades).
# Designed for low trade frequency (target: 30-100 total trades over 4 years) to avoid fee drag.
# Works in bull/bear by using weekly regime filter and Camarilla levels as dynamic support/resistance.
# Uses discrete sizing (±0.25) to minimize fee churn.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_camarilla_pivot_volume_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # 1d HTF data (same as primary timeframe for Camarilla calculation)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for 1d
    # Pivot point = (high + low + close) / 3
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    # H4 = pivot + (range * 1.1/2)
    # H3 = pivot + (range * 1.1/4)
    # L3 = pivot - (range * 1.1/4)
    # L4 = pivot - (range * 1.1/2)
    h3 = pivot + (range_1d * 1.1 / 4)
    l3 = pivot - (range_1d * 1.1 / 4)
    h4 = pivot + (range_1d * 1.1 / 2)
    l4 = pivot - (range_1d * 1.1 / 2)
    
    # Align Camarilla levels to 1d timeframe (completed 1d candle only)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    
    # 1w HTF data for choppiness index regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate True Range for 1w
    tr1 = pd.Series(high_1w).shift(1) - pd.Series(low_1w).shift(1)
    tr2 = abs(pd.Series(high_1w) - pd.Series(close_1w).shift(1))
    tr3 = abs(pd.Series(low_1w) - pd.Series(close_1w).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1w = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate Choppiness Index (14-period)
    # SUM(ATR, 14) / (MAX(HIGH,14) - MIN(LOW,14)) * 100
    sum_atr_14 = pd.Series(atr_1w).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    chop_denominator = max_high_14 - min_low_14
    chop_denominator = np.where(chop_denominator == 0, 1e-10, chop_denominator)
    chop_1w = (sum_atr_14 / chop_denominator) * 100
    chop_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_1w)
    
    # Volume spike detection (20-period volume average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma_20 * 2.0)  # Volume at least 2x average
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or
            np.isnan(chop_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Weekly regime filter: only trade in trending markets (CHOP <= 61.8)
        trending = chop_1w_aligned[i] <= 61.8
        
        if position == 1:  # Long position
            # Exit: price falls below L3 level
            if close[i] < l3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price rises above H3 level
            if close[i] > h3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above H3 level with volume spike
            if (close[i] > h3_aligned[i]) and \
               (vol_spike[i]) and \
               (trending):
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below L3 level with volume spike
            elif (close[i] < l3_aligned[i]) and \
                 (vol_spike[i]) and \
                 (trending):
                position = -1
                signals[i] = -0.25
    
    return signals