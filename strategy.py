#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray + 1d ADX regime filter with volume confirmation.
- Bull Power = High - EMA13(close), Bear Power = EMA13(close) - Low
- Enter long when Bull Power > 0 and Bear Power < 0 (bullish imbalance) and ADX(14) > 25 (trending)
- Enter short when Bear Power > 0 and Bull Power < 0 (bearish imbalance) and ADX(14) > 25
- Volume confirmation: current volume > 1.5x 20-bar average to filter weak breakouts
- Exit when power imbalance reverses or ADX < 20 (trend weakens)
- Uses 6h timeframe to capture medium-term swings in both bull and bear markets
- Designed to avoid whipsaws in ranging markets via ADX regime filter
- Targets 12-30 trades/year (50-120 total over 4 years) to stay fee-efficient
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
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d EMA13 for Elder Ray calculation
    close_1d = df_1d['close'].values
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # 1d ADX calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    
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
    if len(tr) >= tr_period:
        tr_sum[tr_period-1] = np.nansum(tr[1:tr_period])
        dm_plus_sum[tr_period-1] = np.nansum(dm_plus[1:tr_period])
        dm_minus_sum[tr_period-1] = np.nansum(dm_minus[1:tr_period])
    
    # Wilder's smoothing
    for i in range(tr_period, len(tr)):
        tr_sum[i] = tr_sum[i-1] - (tr_sum[i-1] / tr_period) + tr[i]
        dm_plus_sum[i] = dm_plus_sum[i-1] - (dm_plus_sum[i-1] / tr_period) + dm_plus[i]
        dm_minus_sum[i] = dm_minus_sum[i-1] - (dm_minus_sum[i-1] / tr_period) + dm_minus[i]
    
    # DI+ and DI-
    di_plus = np.where(tr_sum != 0, 100 * dm_plus_sum / tr_sum, 0)
    di_minus = np.where(tr_sum != 0, 100 * dm_minus_sum / tr_sum, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    
    adx = np.full_like(dx, np.nan)
    if len(dx) >= tr_period:
        adx[tr_period-1] = np.nanmean(dx[tr_period:2*tr_period])
        for i in range(2*tr_period, len(dx)):
            adx[i] = (adx[i-1] * (tr_period-1) + dx[i]) / tr_period
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
    bull_power = high_1d - ema_13_1d
    bear_power = ema_13_1d - low_1d
    
    # Align HTF indicators to LTF
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: Bull Power > 0 AND Bear Power < 0 (bullish imbalance) AND ADX > 25 AND volume confirmation
            if bull_power_aligned[i] > 0 and bear_power_aligned[i] < 0 and adx_aligned[i] > 25 and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power > 0 AND Bull Power < 0 (bearish imbalance) AND ADX > 25 AND volume confirmation
            elif bear_power_aligned[i] > 0 and bull_power_aligned[i] < 0 and adx_aligned[i] > 25 and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Power imbalance reverses OR ADX < 20 (trend weakens)
            if bull_power_aligned[i] <= 0 or bear_power_aligned[i] >= 0 or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Power imbalance reverses OR ADX < 20 (trend weakens)
            if bear_power_aligned[i] <= 0 or bull_power_aligned[i] >= 0 or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_ADX_Regime_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0