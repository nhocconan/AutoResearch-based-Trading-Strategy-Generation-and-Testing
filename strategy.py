#!/usr/bin/env python3
# 4h_1d_Camarilla_R1_S1_Breakout_Volume_Trend_Filter
# Hypothesis: Camarilla pivot levels (R1/S1) from 1d timeframe act as strong intraday support/resistance.
# Breakouts above R1 or below S1 with volume confirmation and ADX filter capture institutional momentum.
# Works in both bull and bear markets by filtering for strong trends (ADX>25) and avoiding chop.
# Target: 20-50 trades/year to minimize fee drag.

name = "4h_1d_Camarilla_R1_S1_Breakout_Volume_Trend_Filter"
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
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels (R1, S1) from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_r1 = np.full_like(high_1d, np.nan)
    camarilla_s1 = np.full_like(high_1d, np.nan)
    
    for i in range(len(high_1d)):
        if i >= 1:  # Need previous day's data
            # Camarilla formulas
            camarilla_r1[i] = close_1d[i-1] + (high_1d[i-1] - low_1d[i-1]) * 1.1 / 12
            camarilla_s1[i] = close_1d[i-1] - (high_1d[i-1] - low_1d[i-1]) * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Calculate ADX (14-period) for trend strength on 4h data
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
    
    # Smooth TR and DM (14-period)
    tr_sum = np.full_like(high, np.nan)
    dm_plus_sum = np.full_like(high, np.nan)
    dm_minus_sum = np.full_like(high, np.nan)
    
    for i in range(len(high)):
        if i >= 13:  # 14-period smoothing
            tr_sum[i] = np.nansum(tr[i-13:i+1])
            dm_plus_sum[i] = np.nansum(dm_plus[i-13:i+1])
            dm_minus_sum[i] = np.nansum(dm_minus[i-13:i+1])
    
    # Directional Indicators
    di_plus = np.full_like(high, np.nan)
    di_minus = np.full_like(high, np.nan)
    dx = np.full_like(high, np.nan)
    
    valid = ~np.isnan(tr_sum) & (tr_sum != 0)
    di_plus[valid] = 100 * dm_plus_sum[valid] / tr_sum[valid]
    di_minus[valid] = 100 * dm_minus_sum[valid] / tr_sum[valid]
    dx[valid] = 100 * np.abs(di_plus[valid] - di_minus[valid]) / (di_plus[valid] + di_minus[valid])
    
    # ADX (smoothed DX)
    adx = np.full_like(high, np.nan)
    for i in range(len(high)):
        if i >= 27:  # 14 + 13 for ADX smoothing
            valid_dx = dx[i-13:i+1]
            if not np.all(np.isnan(valid_dx)):
                adx[i] = np.nanmean(valid_dx)
    
    # Volume confirmation: volume > 1.3x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma20 * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(28, 1)  # Ensure ADX and Camarilla are calculated
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(adx[i]) or np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla R1 + ADX > 25 + volume confirmation
            if close[i] > camarilla_r1_aligned[i] and adx[i] > 25 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S1 + ADX > 25 + volume confirmation
            elif close[i] < camarilla_s1_aligned[i] and adx[i] > 25 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below Camarilla S1 or ADX weakens
            if close[i] < camarilla_s1_aligned[i] or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above Camarilla R1 or ADX weakens
            if close[i] > camarilla_r1_aligned[i] or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals