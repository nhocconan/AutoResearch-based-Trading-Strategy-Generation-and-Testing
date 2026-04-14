#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1-week Pivot Point levels (from weekly high/low/close) with
# volume confirmation and ADX trend filter. Long when price breaks above weekly pivot (R1) with
# volume > 1.5x average and ADX > 25. Short when price breaks below weekly pivot (S1) with
# volume > 1.5x average and ADX > 25. Exit when price returns to weekly pivot or ADX drops below 20.
# Uses weekly structure for direction, volume for confirmation, ADX for trend strength.
# Designed to work in both bull (breakouts) and bear (breakdowns) markets.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:  # Need at least one complete week
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points (standard formula)
    # Pivot = (H + L + C) / 3
    # R1 = 2*Pivot - L
    # S1 = 2*Pivot - H
    pivot = (high_1w + low_1w + close_1w) / 3.0
    r1 = 2 * pivot - low_1w
    s1 = 2 * pivot - high_1w
    
    # Calculate 1-week average volume for confirmation
    vol_1w = df_1w['volume'].values
    avg_vol_1w = np.full_like(vol_1w, np.nan)
    for i in range(len(vol_1w)):
        if i >= 1:  # Need at least one prior week for average
            avg_vol_1w[i] = np.mean(vol_1w[max(0, i-4):i])  # 4-week average
    
    # Align weekly levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    avg_vol_1w_aligned = align_htf_to_ltf(prices, df_1w, avg_vol_1w)
    
    # Load 1d data for ADX (trend filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX (14) - Wilder's smoothing
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR (14) - Wilder's smoothing
    atr = np.full_like(tr, np.nan)
    if len(tr) >= 14:
        atr[13] = np.nanmean(tr[1:14])  # First ATR: simple average
        for i in range(14, len(tr)):
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values with Wilder's smoothing
    tr_14 = np.full_like(tr, np.nan)
    dm_plus_14 = np.full_like(tr, np.nan)
    dm_minus_14 = np.full_like(tr, np.nan)
    
    if len(tr) >= 14:
        tr_14[13] = np.nansum(tr[1:14])
        dm_plus_14[13] = np.nansum(dm_plus[1:14])
        dm_minus_14[13] = np.nansum(dm_minus[1:14])
        for i in range(14, len(tr)):
            tr_14[i] = tr_14[i-1] - (tr_14[i-1] / 14) + tr[i]
            dm_plus_14[i] = dm_plus_14[i-1] - (dm_plus_14[i-1] / 14) + dm_plus[i]
            dm_minus_14[i] = dm_minus_14[i-1] - (dm_minus_14[i-1] / 14) + dm_minus[i]
    
    # DI+ and DI-
    di_plus = np.full_like(tr, np.nan)
    di_minus = np.full_like(tr, np.nan)
    valid = (tr_14 != 0) & ~np.isnan(tr_14)
    di_plus[valid] = 100 * dm_plus_14[valid] / tr_14[valid]
    di_minus[valid] = 100 * dm_minus_14[valid] / tr_14[valid]
    
    # DX and ADX
    dx = np.full_like(tr, np.nan)
    dx_valid = (di_plus + di_minus) != 0
    dx[dx_valid] = 100 * np.abs(di_plus[dx_valid] - di_minus[dx_valid]) / (di_plus[dx_valid] + di_minus[dx_valid])
    
    adx = np.full_like(tr, np.nan)
    if len(dx) >= 14:
        # First ADX: simple average of first 14 DX
        valid_14 = ~np.isnan(dx[1:15])
        if np.sum(valid_14) >= 14:
            adx[14] = np.nanmean(dx[1:15])
        for i in range(15, len(dx)):
            if not np.isnan(dx[i-1]) and not np.isnan(dx[i]):
                adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(34, 14)  # Need ADX period
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(pivot_aligned[i]) or 
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(avg_vol_1w_aligned[i]) or
            np.isnan(adx_aligned[i]) or
            np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x weekly average
        volume_confirmed = volume[i] > 1.5 * avg_vol_1w_aligned[i]
        
        # Trend filter: ADX > 25 indicates strong trend
        strong_trend = adx_aligned[i] > 25
        
        # Weak trend filter: ADX < 20 indicates trend weakening
        weak_trend = adx_aligned[i] < 20
        
        if position == 0:
            # Look for breakout entries in strong trend with volume
            # Long: price breaks above R1 with volume and strong trend
            if (close[i] > r1_aligned[i] and 
                volume_confirmed and
                strong_trend):
                position = 1
                signals[i] = position_size
            # Short: price breaks below S1 with volume and strong trend
            elif (close[i] < s1_aligned[i] and 
                  volume_confirmed and
                  strong_trend):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to pivot or trend weakens
            if (close[i] <= pivot_aligned[i] or 
                weak_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to pivot or trend weakens
            if (close[i] >= pivot_aligned[i] or 
                weak_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1w_Pivot_R1S1_Volume_ADX_Filter_v1"
timeframe = "6h"
leverage = 1.0