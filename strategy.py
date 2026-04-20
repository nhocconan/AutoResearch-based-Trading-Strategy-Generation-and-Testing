#!/usr/bin/env python3
# 4h_Donchian20_Breakout_Volume_Confirm_ADXFilter
# Hypothesis: Donchian(20) breakouts capture momentum, volume > 1.5x MA confirms institutional participation,
# and ADX(14) > 20 filters choppy markets. Works in both bull and bear markets by taking breakouts in either direction.
# Target: 150-250 total trades over 4 years (38-63/year) to balance opportunity and fee drag.

name = "4h_Donchian20_Breakout_Volume_Confirm_ADXFilter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get ADX from 1h timeframe for regime filter
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 30:
        return np.zeros(n)
    
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    close_1h = df_1h['close'].values
    
    # Calculate True Range
    tr1 = np.abs(high_1h[1:] - low_1h[1:])
    tr2 = np.abs(high_1h[1:] - close_1h[:-1])
    tr3 = np.abs(low_1h[1:] - close_1h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # Calculate +DM and -DM
    up_move = high_1h[1:] - high_1h[:-1]
    down_move = low_1h[:-1] - low_1h[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
    atr = np.full_like(tr, np.nan)
    plus_dm_smooth = np.full_like(plus_dm, np.nan)
    minus_dm_smooth = np.full_like(minus_dm, np.nan)
    
    period = 14
    alpha = 1.0 / period
    
    # Initialize first value with simple average
    if len(tr) >= period:
        atr[period-1] = np.nanmean(tr[1:period])
        plus_dm_smooth[period-1] = np.nanmean(plus_dm[1:period])
        minus_dm_smooth[period-1] = np.nanmean(minus_dm[1:period])
        
        # Wilder smoothing
        for i in range(period, len(tr)):
            atr[i] = alpha * tr[i] + (1 - alpha) * atr[i-1]
            plus_dm_smooth[i] = alpha * plus_dm[i] + (1 - alpha) * plus_dm_smooth[i-1]
            minus_dm_smooth[i] = alpha * minus_dm[i] + (1 - alpha) * minus_dm_smooth[i-1]
    
    # Calculate DI and DX
    plus_di = np.full_like(tr, np.nan)
    minus_di = np.full_like(tr, np.nan)
    dx = np.full_like(tr, np.nan)
    
    valid = ~np.isnan(atr) & (atr > 0)
    plus_di[valid] = 100 * plus_dm_smooth[valid] / atr[valid]
    minus_di[valid] = 100 * minus_dm_smooth[valid] / atr[valid]
    
    di_sum = plus_di + minus_di
    valid_dx = (di_sum > 0) & ~np.isnan(di_sum)
    dx[valid_dx] = 100 * np.abs(plus_di[valid_dx] - minus_di[valid_dx]) / di_sum[valid_dx]
    
    # ADX = smoothed DX
    adx = np.full_like(dx, np.nan)
    adx_period = 14
    alpha_adx = 1.0 / adx_period
    
    # Initialize ADX
    valid_dx_start = np.where(~np.isnan(dx))[0]
    if len(valid_dx_start) >= adx_period:
        start_idx = valid_dx_start[adx_period-1]
        adx[start_idx] = np.nanmean(dx[valid_dx_start[:adx_period]])
        
        for i in range(start_idx + 1, len(dx)):
            if not np.isnan(dx[i]):
                adx[i] = alpha_adx * dx[i] + (1 - alpha_adx) * adx[i-1]
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1h, adx)
    
    # Donchian(20) channels on 4h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above Donchian high with volume confirmation and ADX > 20
            if (close[i] > highest_high[i] and volume_confirm[i] and adx_aligned[i] > 20):
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low with volume confirmation and ADX > 20
            elif (close[i] < lowest_low[i] and volume_confirm[i] and adx_aligned[i] > 20):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below Donchian low (reversal) or ADX drops
            if close[i] < lowest_low[i] or adx_aligned[i] < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above Donchian high (reversal) or ADX drops
            if close[i] > highest_high[i] or adx_aligned[i] < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals