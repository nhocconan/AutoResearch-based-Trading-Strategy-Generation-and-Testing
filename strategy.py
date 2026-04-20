#!/usr/bin/env python3
# 6h_WeeklyPivot_Donchian_Breakout_Momentum
# Hypothesis: Weekly pivot levels (from Monday) act as key support/resistance for 6h trends.
# Breakouts above weekly pivot + R1 with volume/ADX confirmation capture momentum.
# Works in bull markets (breakouts continue) and bear markets (rejections at resistance).
# Weekly pivot provides structural context; 6h timeframe reduces noise vs lower TFs.
# Target: 15-35 trades/year (60-140 total over 4 years) to minimize fee drag.

name = "6h_WeeklyPivot_Donchian_Breakout_Momentum"
timeframe = "6h"
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
    
    # Get weekly data for pivot calculation (once before loop)
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 10:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week's OHLC)
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    r1 = 2 * pivot - weekly_low
    s1 = 2 * pivot - weekly_high
    r2 = pivot + (weekly_high - weekly_low)
    s2 = pivot - (weekly_high - weekly_low)
    r3 = weekly_high + 2 * (pivot - weekly_low)
    s3 = weekly_low - 2 * (weekly_high - pivot)
    
    # Align weekly levels to 6h (wait for weekly bar to close)
    pivot_6h = align_htf_to_ltf(prices, df_weekly, pivot)
    r1_6h = align_htf_to_ltf(prices, df_weekly, r1)
    s1_6h = align_htf_to_ltf(prices, df_weekly, s1)
    r2_6h = align_htf_to_ltf(prices, df_weekly, r2)
    s2_6h = align_htf_to_ltf(prices, df_weekly, s2)
    r3_6h = align_htf_to_ltf(prices, df_weekly, r3)
    s3_6h = align_htf_to_ltf(prices, df_weekly, s3)
    
    # Get 6h data for Donchian channels (20-period)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    # Calculate Donchian channels (20-period)
    donchian_high = np.full_like(high_6h, np.nan)
    donchian_low = np.full_like(high_6h, np.nan)
    
    for i in range(len(high_6h)):
        if i >= 19:  # 20-period lookback
            donchian_high[i] = np.max(high_6h[i-19:i+1])
            donchian_low[i] = np.min(low_6h[i-19:i+1])
    
    # Align Donchian levels to 6h
    donchian_high_6h = align_htf_to_ltf(prices, df_6h, donchian_high)
    donchian_low_6h = align_htf_to_ltf(prices, df_6h, donchian_low)
    
    # ADX (14-period) for trend strength
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
    
    # ADX (smoothed DX, 14-period)
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
    
    start_idx = max(28, 20)  # Ensure ADX and Donchian are calculated
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_6h[i]) or np.isnan(r1_6h[i]) or np.isnan(s1_6h[i]) or
            np.isnan(donchian_high_6h[i]) or np.isnan(donchian_low_6h[i]) or
            np.isnan(adx[i]) or np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above weekly R1 + Donchian high + ADX > 25 + volume
            if (close[i] > r1_6h[i] and close[i] > donchian_high_6h[i] and 
                adx[i] > 25 and volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S1 + Donchian low + ADX > 25 + volume
            elif (close[i] < s1_6h[i] and close[i] < donchian_low_6h[i] and 
                  adx[i] > 25 and volume_filter[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below weekly pivot or Donchian low or ADX weakens
            if (close[i] < pivot_6h[i] or close[i] < donchian_low_6h[i] or adx[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above weekly pivot or Donchian high or ADX weakens
            if (close[i] > pivot_6h[i] or close[i] > donchian_high_6h[i] or adx[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals