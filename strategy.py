#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray with 1d ADX regime filter and volume confirmation.
- Bull Power = High - EMA13(close), Bear Power = EMA13(close) - Low
- Long when Bull Power > 0 AND Bear Power < 0 AND 1d ADX > 25 (trending market) AND volume > 1.5 * median volume
- Short when Bear Power > 0 AND Bull Power < 0 AND 1d ADX > 25 (trending market) AND volume > 1.5 * median volume
- Exit when power signals weaken or ADX drops below 20 (range market) or volume normalizes
- Uses 6h primary timeframe with 1d HTF to target 50-150 total trades over 4 years (12-37/year)
- Elder Ray measures bull/bear power relative to EMA, effective in both bull and bear markets
- 1d ADX regime filter ensures we only trade in trending conditions, avoiding whipsaws in ranges
- Volume confirmation reduces false signals
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
    
    # Calculate EMA13 for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
    bull_power = high - ema_13
    bear_power = ema_13 - low
    
    # Get 1d data ONCE before loop for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    # Calculate 1d ADX (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
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
    
    # Initial values
    tr_sum[tr_period] = np.nansum(tr[1:tr_period+1])
    dm_plus_sum[tr_period] = np.nansum(dm_plus[1:tr_period+1])
    dm_minus_sum[tr_period] = np.nansum(dm_minus[1:tr_period+1])
    
    # Wilder's smoothing
    for i in range(tr_period + 1, len(tr)):
        tr_sum[i] = tr_sum[i-1] - (tr_sum[i-1] / tr_period) + tr[i]
        dm_plus_sum[i] = dm_plus_sum[i-1] - (dm_plus_sum[i-1] / tr_period) + dm_plus[i]
        dm_minus_sum[i] = dm_minus_sum[i-1] - (dm_minus_sum[i-1] / tr_period) + dm_minus[i]
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_sum / tr_sum
    di_minus = 100 * dm_minus_sum / tr_sum
    
    # DX and ADX
    dx = np.zeros_like(di_plus)
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    
    # ADX (smoothed DX)
    adx_14 = np.zeros_like(dx)
    adx_14[2*tr_period] = np.nanmean(dx[tr_period+1:2*tr_period+1])
    
    for i in range(2*tr_period + 1, len(dx)):
        adx_14[i] = (adx_14[i-1] * (tr_period - 1) + dx[i]) / tr_period
    
    # Align 1d ADX to 6h timeframe
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # Volume confirmation: volume > 1.5 * median volume of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    volume_confirmed = volume > (1.5 * vol_median)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(30, 2*14+1) + 5  # ADX needs warmup + buffer
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(adx_14_aligned[i]) or np.isnan(vol_median[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        adx_val = adx_14_aligned[i]
        
        if position == 0:
            # Long: Bull Power > 0, Bear Power < 0, ADX > 25 (trending), volume confirmed
            if bull_power[i] > 0 and bear_power[i] < 0 and adx_val > 25 and volume_confirmed[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power > 0, Bull Power < 0, ADX > 25 (trending), volume confirmed
            elif bear_power[i] > 0 and bull_power[i] < 0 and adx_val > 25 and volume_confirmed[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Power weakens OR ADX drops below 20 (range) OR volume drops
            if bull_power[i] <= 0 or bear_power[i] >= 0 or adx_val < 20 or not volume_confirmed[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Power weakens OR ADX drops below 20 (range) OR volume drops
            if bear_power[i] <= 0 or bull_power[i] >= 0 or adx_val < 20 or not volume_confirmed[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_1dADX_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0