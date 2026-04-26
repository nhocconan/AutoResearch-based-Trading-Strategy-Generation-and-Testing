#!/usr/bin/env python3
"""
6h_WeeklyPivot_Donchian20_Breakout_1dTrend_VolumeSpike
Hypothesis: Combines weekly pivot levels (from 1w HTF) as structure with 6h Donchian(20) breakouts, filtered by 1d EMA50 trend and volume confirmation. 
Weekly pivots provide significant support/resistance that price respects across market cycles. 
Donchian breakouts capture momentum when price breaks weekly pivot-defined ranges. 
Volume spike confirms institutional participation. 
1d EMA50 filter ensures we trade with the higher timeframe trend, reducing whipsaws in ranging markets. 
Designed for 50-150 total trades over 4 years by requiring confluence of weekly pivot proximity, Donchian breakout, trend alignment, and volume. 
Works in bull/bear markets via 1d trend filter and weekly pivot structure that adapts to changing volatility regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly pivot points (standard formula)
    # P = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    r2_1w = pivot_1w + (high_1w - low_1w)
    s2_1w = pivot_1w - (high_1w - low_1w)
    r3_1w = high_1w + 2 * (pivot_1w - low_1w)
    s3_1w = low_1w - 2 * (high_1w - pivot_1w)
    
    # Align weekly pivot levels to 6h timeframe (wait for weekly bar close)
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 6h Donchian(20) channels
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume spike: volume > 1.5x 20-period median volume
    volume_series = pd.Series(volume)
    vol_median_20 = volume_series.rolling(window=20, min_periods=20).median().values
    volume_spike = volume > (1.5 * vol_median_20)
    
    # Position sizing: 0.25 (25% of capital) to balance opportunity and risk
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need 50 for weekly/1d indicators, 20 for Donchian, 20 for volume median
    start_idx = max(50, 20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(pivot_1w_aligned[i]) or
            np.isnan(r1_1w_aligned[i]) or
            np.isnan(s1_1w_aligned[i]) or
            np.isnan(r2_1w_aligned[i]) or
            np.isnan(s2_1w_aligned[i]) or
            np.isnan(r3_1w_aligned[i]) or
            np.isnan(s3_1w_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i]) or
            np.isnan(vol_median_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        ema_50_val = ema_50_1d_aligned[i]
        vol_spike = volume_spike[i]
        size = fixed_size
        
        if position == 0:
            # Flat - look for entry
            # Long conditions: price near weekly support (S1/S2), breaks above Donchian high, uptrend, volume spike
            near_support = (low_val <= s1_1w_aligned[i] * 1.02) or (low_val <= s2_1w_aligned[i] * 1.02)
            donchian_breakout = high_val > highest_high[i]
            uptrend = close_val > ema_50_val
            
            long_entry = near_support and donchian_breakout and uptrend and vol_spike
            
            # Short conditions: price near weekly resistance (R1/R2), breaks below Donchian low, downtrend, volume spike
            near_resistance = (high_val >= r1_1w_aligned[i] * 0.98) or (high_val >= r2_1w_aligned[i] * 0.98)
            donchian_breakdown = low_val < lowest_low[i]
            downtrend = close_val < ema_50_val
            
            short_entry = near_resistance and donchian_breakdown and downtrend and vol_spike
            
            if long_entry:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_entry:
                signals[i] = -size
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on trend reversal, Donchian breakdown, or at weekly resistance
            if close_val < ema_50_val or low_val < lowest_low[i] or high_val >= r1_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on trend reversal, Donchian breakout, or at weekly support
            if close_val > ema_50_val or high_val > highest_high[i] or low_val <= s1_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "6h_WeeklyPivot_Donchian20_Breakout_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0