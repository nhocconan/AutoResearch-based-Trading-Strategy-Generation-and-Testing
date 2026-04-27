#!/usr/bin/env python3
"""
12h CAMARILLA PIVOT BREAKOUT + DAILY VOLUME SPIKE + WEEKLY CHOPPINESS FILTER
Hypothesis: Price breaking above/below CAMARILLA R3/S3 levels on 12h with daily volume
spike and weekly choppy market (range) conditions produces high-probability mean-reversion
entries. Uses weekly chop to avoid trending markets where breakouts fail. Works in bull
and bear markets by adapting to range conditions via chop filter. Target: 15-30 trades/year.
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
    
    # Get daily data for CAMARILLA pivot and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Get weekly data for chop filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate DAILY CAMARILLA pivot levels (R3, S3)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    R3_1d = pivot_1d + range_1d * 1.1
    S3_1d = pivot_1d - range_1d * 1.1
    
    # Calculate DAILY volume spike filter (volume > 2x 20-day average)
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = vol_1d > (2.0 * vol_ma_20_1d)
    
    # Calculate WEEKLY chop filter (Choppiness Index > 61.8 = ranging)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum.reduce([tr1, tr2, tr3])
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    
    # ATR(14)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of TRUE RANGE over 14 periods
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log(sum_tr_14 / (hh_14 - ll_14)) / log(14)
    range_14 = hh_14 - ll_14
    chop = 100 * np.log(sum_tr_14 / range_14) / np.log(14)
    chop = np.where(range_14 == 0, 100, chop)  # avoid div by zero
    
    # Choppiness > 61.8 indicates ranging market
    chop_filter = chop > 61.8
    
    # Align HTF arrays to 12h timeframe
    R3_1d_aligned = align_htf_to_ltf(prices, df_1d, R3_1d)
    S3_1d_aligned = align_htf_to_ltf(prices, df_1d, S3_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d.astype(float))
    chop_filter_aligned = align_htf_to_ltf(prices, df_1w, chop_filter.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position
    
    # Warmup: need daily pivot (5), volume MA (20), weekly chop (14)
    start_idx = max(5, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(R3_1d_aligned[i]) or np.isnan(S3_1d_aligned[i]) or
            np.isnan(vol_spike_1d_aligned[i]) or np.isnan(chop_filter_aligned[i])):
            signals[i] = 0.0
            continue
        
        price_now = close[i]
        vol_spike = bool(vol_spike_1d_aligned[i])
        chop_filter_val = bool(chop_filter_aligned[i])
        R3 = R3_1d_aligned[i]
        S3 = S3_1d_aligned[i]
        
        # Breakout conditions
        breakout_up = price_now > R3
        breakout_down = price_now < S3
        
        # Entry conditions: breakout + volume spike + choppy (ranging) market
        if position == 0:
            if breakout_up and vol_spike and chop_filter_val:
                # In ranging market, selling the breakout (fade)
                signals[i] = -size
                position = -1
            elif breakout_down and vol_spike and chop_filter_val:
                # In ranging market, buying the breakout (fade)
                signals[i] = size
                position = 1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to pivot or volatility expands
            if price_now < pivot_1d[-1] or not chop_filter_val:  # pivot from last available day
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to pivot or volatility expands
            if price_now > pivot_1d[-1] or not chop_filter_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_R3S3_Breakout_1dVolumeSpike_1wChop"
timeframe = "12h"
leverage = 1.0