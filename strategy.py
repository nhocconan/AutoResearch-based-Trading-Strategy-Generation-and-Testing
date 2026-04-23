#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d ADX(14) regime filter and volume confirmation.
Long when price breaks above 12h Donchian high AND 1d ADX > 25 (trending) AND volume > 1.5x MA20.
Short when price breaks below 12h Donchian low AND 1d ADX > 25 AND volume > 1.5x MA20.
Exit on break of opposite Donchian level or ADX < 20 (range regime).
Discrete position size 0.25 to minimize fee churn. Works in trending markets via ADX filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d ADX(14) for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # first tr is NaN
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR, DM+
    tr_period = 14
    tr_sum = np.zeros_like(tr)
    dm_plus_sum = np.zeros_like(dm_plus)
    dm_minus_sum = np.zeros_like(dm_minus)
    
    # Initial sum (first 14 periods)
    for i in range(1, tr_period + 1):
        if i < len(tr):
            tr_sum[i] = tr_sum[i-1] + tr[i]
            dm_plus_sum[i] = dm_plus_sum[i-1] + dm_plus[i]
            dm_minus_sum[i] = dm_minus_sum[i-1] + dm_minus[i]
    
    # Wilder's smoothing
    for i in range(tr_period + 1, len(tr)):
        tr_sum[i] = tr_sum[i-1] - (tr_sum[i-1] / tr_period) + tr[i]
        dm_plus_sum[i] = dm_plus_sum[i-1] - (dm_plus_sum[i-1] / tr_period) + dm_plus[i]
        dm_minus_sum[i] = dm_minus_sum[i-1] - (dm_minus_sum[i-1] / tr_period) + dm_minus[i]
    
    # DI+ and DI-
    di_plus = np.full_like(tr, np.nan)
    di_minus = np.full_like(tr, np.nan)
    dx = np.full_like(tr, np.nan)
    
    for i in range(tr_period, len(tr)):
        if tr_sum[i] != 0:
            di_plus[i] = 100 * dm_plus_sum[i] / tr_sum[i]
            di_minus[i] = 100 * dm_minus_sum[i] / tr_sum[i]
            dx[i] = 100 * np.abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])
    
    # ADX: smoothed DX
    adx = np.full_like(tr, np.nan)
    adx_period = 14
    
    # Initial ADX (first 14 DX after period)
    dx_valid = dx[~np.isnan(dx)]
    if len(dx_valid) >= adx_period:
        first_adx_idx = np.where(~np.isnan(dx))[0][adx_period-1] if np.sum(~np.isnan(dx)) >= adx_period else len(tr)
        if first_adx_idx < len(tr):
            adx[first_adx_idx] = np.nanmean(dx[first_adx_idx-adx_period+1:first_adx_idx+1])
    
    # Subsequent ADX values
    for i in range(first_adx_idx + 1, len(tr)):
        if not np.isnan(dx[i]):
            prev_adx = adx[i-1] if not np.isnan(adx[i-1]) else 0
            adx[i] = (prev_adx * (adx_period - 1) + dx[i]) / adx_period
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 12h Donchian(20) channels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Donchian high: max(high, 20)
    donch_high_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    # Donchian low: min(low, 20)
    donch_low_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    donch_high_aligned = align_htf_to_ltf(prices, df_12h, donch_high_12h)
    donch_low_aligned = align_htf_to_ltf(prices, df_12h, donch_low_12h)
    
    # Volume MA(20) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # need ADX (34=14+20), Donchian(20), volume MA20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: ADX > 25 = trending, ADX < 20 = range
        trending = adx_aligned[i] > 25
        ranging = adx_aligned[i] < 20
        
        # Volume filter: 12h volume > 1.5x 20-period MA
        vol_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: Break above Donchian high AND trending AND volume confirmation
            if close[i] > donch_high_aligned[i] and trending and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian low AND trending AND volume confirmation
            elif close[i] < donch_low_aligned[i] and trending and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            if position == 1:
                # Exit long on break below Donchian low OR ADX < 20 (range)
                if close[i] < donch_low_aligned[i] or ranging:
                    exit_signal = True
            elif position == -1:
                # Exit short on break above Donchian high OR ADX < 20 (range)
                if close[i] > donch_high_aligned[i] or ranging:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Donchian20_Breakout_1dADX_Regime_VolumeConfirmation"
timeframe = "12h"
leverage = 1.0