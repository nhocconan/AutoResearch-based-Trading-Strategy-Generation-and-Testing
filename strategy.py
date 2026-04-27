#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian breakout (20) with weekly pivot direction filter and volume spike confirmation
# Uses weekly pivot levels to establish bias (only long above weekly pivot, short below) and
# Donchian channel breakouts for entry timing. Volume > 2x 20-period average confirms breakout strength.
# Weekly pivot provides structural bias that works in both bull and bear markets by aligning with
# higher timeframe trend. Target: 25-35 trades/year to minimize fee decay while capturing strong moves.
# Focus on BTC/ETH as primary assets.

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
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot points (standard floor trader pivots)
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    n_1w = len(close_1w)
    
    pivot_1w = np.full(n_1w, np.nan)
    r1_1w = np.full(n_1w, np.nan)
    s1_1w = np.full(n_1w, np.nan)
    r2_1w = np.full(n_1w, np.nan)
    s2_1w = np.full(n_1w, np.nan)
    r3_1w = np.full(n_1w, np.nan)
    s3_1w = np.full(n_1w, np.nan)
    
    for i in range(n_1w):
        if np.isnan(high_1w[i]) or np.isnan(low_1w[i]) or np.isnan(close_1w[i]):
            continue
        pivot = (high_1w[i] + low_1w[i] + close_1w[i]) / 3.0
        pivot_1w[i] = pivot
        r1_1w[i] = 2 * pivot - low_1w[i]
        s1_1w[i] = 2 * pivot - high_1w[i]
        r2_1w[i] = pivot + (high_1w[i] - low_1w[i])
        s2_1w[i] = pivot - (high_1w[i] - low_1w[i])
        r3_1w[i] = high_1w[i] + 2 * (pivot - low_1w[i])
        s3_1w[i] = low_1w[i] - 2 * (high_1w[i] - pivot)
    
    # Align weekly pivot levels to 6h
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    
    # Donchian channel (20-period) on 6h
    dc_period = 20
    upper_dc = np.full(n, np.nan)
    lower_dc = np.full(n, np.nan)
    
    for i in range(dc_period, n):
        upper_dc[i] = np.max(high[i-dc_period:i])
        lower_dc[i] = np.min(low[i-dc_period:i])
    
    # 20-period average volume for spike detection
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # 25% position size
    
    # Warmup period
    start_idx = max(dc_period, vol_period)
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_1w_aligned[i]) or 
            np.isnan(upper_dc[i]) or 
            np.isnan(lower_dc[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Determine bias from weekly pivot
        # Long bias: price above weekly pivot
        # Short bias: price below weekly pivot
        long_bias = price > pivot_1w_aligned[i]
        short_bias = price < pivot_1w_aligned[i]
        
        # Breakout conditions
        breakout_up = price > upper_dc[i]
        breakout_down = price < lower_dc[i]
        
        # Volume confirmation: spike > 2x average
        volume_confirmation = vol_ratio > 2.0
        
        if position == 0:
            # Long: bullish breakout with long bias and volume
            if long_bias and breakout_up and volume_confirmation:
                signals[i] = size
                position = 1
            # Short: bearish breakout with short bias and volume
            elif short_bias and breakout_down and volume_confirmation:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price returns to weekly pivot or breaks below lower Donchian
            if price < pivot_1w_aligned[i] or price < lower_dc[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: price returns to weekly pivot or breaks above upper Donchian
            if price > pivot_1w_aligned[i] or price > upper_dc[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Donchian_WeeklyPivot_Volume"
timeframe = "6h"
leverage = 1.0