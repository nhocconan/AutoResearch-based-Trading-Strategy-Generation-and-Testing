#!/usr/bin/env python3
# 4h_PivotBreakout_VolumeTrend
# Hypothesis: Combines 1D pivot points (classic) with 4H breakout logic and volume confirmation.
# Pivot points provide key support/resistance levels derived from prior day's action.
# A break above R1 or below S1 with volume confirmation indicates institutional interest.
# ADX filter ensures we only trade in trending markets (>25) to avoid chop.
# Works in both bull and bear markets by following the breakout direction.
# Target: 25-40 trades/year (~100-160 total over 4 years) to stay within optimal trade frequency for 4h.

name = "4h_PivotBreakout_VolumeTrend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1D data for pivot points (classic)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot points: P = (H + L + C)/3
    # R1 = 2*P - L, S1 = 2*P - H
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    
    # Align pivot levels to 4H timeframe (using previous day's values for breakout)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 4H ADX (14-period) for trend strength
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    
    # Directional Movement
    dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                       np.maximum(high[1:] - high[:-1], 0), 0)
    dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                        np.maximum(low[:-1] - low[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values with proper Wilder's smoothing (alpha = 1/14)
    tr_ma = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    dm_plus_ma = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False).mean().values
    dm_minus_ma = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False).mean().values
    
    # Avoid division by zero
    tr_ma_safe = np.where(tr_ma == 0, 1e-10, tr_ma)
    
    # DI and DX
    di_plus = 100 * dm_plus_ma / tr_ma_safe
    di_minus = 100 * dm_minus_ma / tr_ma_safe
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    # Volume confirmation: current volume > 1.3 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure sufficient warmup for volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(adx[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1, ADX > 25 (trending), volume confirmation
            if (close[i] > r1_aligned[i] and 
                adx[i] > 25 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1, ADX > 25 (trending), volume confirmation
            elif (close[i] < s1_aligned[i] and 
                  adx[i] > 25 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price returns below pivot (mean reversion) OR ADX drops below 20 (losing trend)
            if (close[i] < pivot_aligned[i] or 
                adx[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price returns above pivot (mean reversion) OR ADX drops below 20 (losing trend)
            if (close[i] > pivot_aligned[i] or 
                adx[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals