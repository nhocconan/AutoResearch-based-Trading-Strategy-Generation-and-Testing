#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R Extreme + 1d ADX Trend + Volume Confirmation
- Williams %R(14) identifies overbought/oversold extremes: < -80 = oversold, > -20 = overbought
- 1d ADX(14) > 25 filters for trending markets (avoids chop/range)
- In trending markets, fade extremes: long when %R < -80, short when %R > -20
- Volume confirmation (>1.5x 20-period average) reduces false signals
- Discrete position sizing (0.25) manages drawdown in 2022 crash
- Target: 60-120 total trades over 4 years (15-30/year) on 6h timeframe
- Works in both bull/bear via ADX trend filter + mean reversion in trends
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
    
    # Get 1d data ONCE before loop for ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d ADX(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align length
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR, DM+,
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
    di_plus = np.where(tr_sum != 0, 100 * dm_plus_sum / tr_sum, 0)
    di_minus = np.where(tr_sum != 0, 100 * dm_minus_sum / tr_sum, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = np.zeros_like(dx)
    # Initial ADX (first 14 DX values after period)
    for i in range(tr_period, min(2*tr_period, len(dx))):
        if i < len(dx):
            adx[i] = np.mean(dx[tr_period:i+1])
    # Wilder's smoothing for ADX
    for i in range(2*tr_period, len(dx)):
        adx[i] = (adx[i-1] * (tr_period - 1) + dx[i]) / tr_period
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Williams %R(14) on 6h
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Only trade in trending markets (ADX > 25)
        if adx_aligned[i] > 25:
            if position == 0:
                # Long: Williams %R oversold (< -80) with volume confirmation
                if williams_r[i] < -80 and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: Williams %R overbought (> -20) with volume confirmation
                elif williams_r[i] > -20 and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
            elif position == 1:
                # Long exit: Williams %R returns above -50 (mean reversion)
                if williams_r[i] > -50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Short exit: Williams %R returns below -50
                if williams_r[i] < -50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
        else:
            # In ranging markets (ADX <= 25), stay flat
            if position != 0:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WilliamsR_Extreme_1dADX_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0