#!/usr/bin/env python3
# 4h_HighLow_Momentum_Reversal
# Hypothesis: Reversals occur when price moves beyond recent high/low ranges with volume confirmation.
# Uses recent high/low breakouts filtered by volume spike and ADX trend strength.
# Works in both bull and bear markets by capturing mean-reversion moves after overextensions.
# Target: 20-50 trades/year.

name = "4h_HighLow_Momentum_Reversal"
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
    
    # Get 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate recent high/low (10-period) for mean reversion signals
    recent_high = np.full_like(high_4h, np.nan)
    recent_low = np.full_like(low_4h, np.nan)
    
    for i in range(len(high_4h)):
        if i >= 9:  # 10-period lookback
            recent_high[i] = np.max(high_4h[i-9:i+1])
            recent_low[i] = np.min(low_4h[i-9:i+1])
    
    # Align recent high/low to LTF
    recent_high_aligned = align_htf_to_ltf(prices, df_4h, recent_high)
    recent_low_aligned = align_htf_to_ltf(prices, df_4h, recent_low)
    
    # Calculate ADX (14-period) for trend strength filter
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
    
    # Smooth TR and DM
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
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(28, 20)  # Ensure ADX and recent high/low are calculated
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(recent_high_aligned[i]) or np.isnan(recent_low_aligned[i]) or
            np.isnan(adx[i]) or np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks below recent low (oversold) + ADX < 30 (weak trend) + volume confirmation
            if close[i] < recent_low_aligned[i] and adx[i] < 30 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks above recent high (overbought) + ADX < 30 (weak trend) + volume confirmation
            elif close[i] > recent_high_aligned[i] and adx[i] < 30 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks above recent high or ADX strengthens
            if close[i] > recent_high_aligned[i] or adx[i] > 35:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks below recent low or ADX strengthens
            if close[i] < recent_low_aligned[i] or adx[i] > 35:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals