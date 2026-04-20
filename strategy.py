#!/usr/bin/env python3
# 4h_Camarilla_Pivot_R1S1_Breakout_VolumeTrend
# Hypothesis: Camarilla pivot levels (R1, S1) from daily timeframe act as strong support/resistance.
# Price breaking these levels with volume confirmation and ADX > 25 indicates institutional interest.
# Works in bull/bear markets as breaks often signal continuation. Position size 0.25.
# Target: 20-40 trades/year (80-160 total) to minimize fee drag.

name = "4h_Camarilla_Pivot_R1S1_Breakout_VolumeTrend"
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
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each daily bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla formulas
    R1 = close_1d + 1.1 * (high_1d - low_1d) / 12
    S1 = close_1d - 1.1 * (high_1d - low_1d) / 12
    
    # Align to 4h timeframe (wait for daily bar to close)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Calculate ADX (14-period) for trend strength on 4h
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                       np.maximum(high[1:] - high[:-1], 0), 0)
    dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                        np.maximum(low[:-1] - low[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smooth TR and DM with Wilder's smoothing (using EMA-like approach)
    tr_sum = np.full_like(high, np.nan)
    dm_plus_sum = np.full_like(high, np.nan)
    dm_minus_sum = np.full_like(high, np.nan)
    
    # Wilder smoothing: first value is sum, then each new value = prev - (prev/14) + current
    tr_sum[13] = np.nansum(tr[0:14])
    dm_plus_sum[13] = np.nansum(dm_plus[0:14])
    dm_minus_sum[13] = np.nansum(dm_minus[0:14])
    
    for i in range(14, len(high)):
        tr_sum[i] = tr_sum[i-1] - (tr_sum[i-1] / 14) + tr[i]
        dm_plus_sum[i] = dm_plus_sum[i-1] - (dm_plus_sum[i-1] / 14) + dm_plus[i]
        dm_minus_sum[i] = dm_minus_sum[i-1] - (dm_minus_sum[i-1] / 14) + dm_minus[i]
    
    # Directional Indicators
    di_plus = np.full_like(high, np.nan)
    di_minus = np.full_like(high, np.nan)
    dx = np.full_like(high, np.nan)
    
    valid = ~np.isnan(tr_sum) & (tr_sum != 0)
    di_plus[valid] = 100 * dm_plus_sum[valid] / tr_sum[valid]
    di_minus[valid] = 100 * dm_minus_sum[valid] / tr_sum[valid]
    dx[valid] = 100 * np.abs(di_plus[valid] - di_minus[valid]) / (di_plus[valid] + di_minus[valid])
    
    # ADX (smoothed DX with Wilder smoothing)
    adx = np.full_like(high, np.nan)
    valid_dx = ~np.isnan(dx)
    if np.any(valid_dx):
        adx[27] = np.nanmean(dx[14:28])  # First ADX value after DX period
        for i in range(28, len(high)):
            adx[i] = adx[i-1] - (adx[i-1] / 14) + dx[i]
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(28, 20)  # Ensure ADX and volume MA are calculated
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or
            np.isnan(adx[i]) or np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 + ADX > 25 + volume confirmation
            if close[i] > R1_aligned[i] and adx[i] > 25 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + ADX > 25 + volume confirmation
            elif close[i] < S1_aligned[i] and adx[i] > 25 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below S1 or ADX weakens significantly
            if close[i] < S1_aligned[i] or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above R1 or ADX weakens significantly
            if close[i] > R1_aligned[i] or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals