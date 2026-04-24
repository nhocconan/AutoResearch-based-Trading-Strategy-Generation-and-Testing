#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian breakout with 1d volume confirmation and 1w ADX trend filter.
- Long when price breaks above Donchian(20) upper band AND 1d volume > 1.5x 20-period average AND 1w ADX > 25
- Short when price breaks below Donchian(20) lower band AND 1d volume > 1.5x 20-period average AND 1w ADX > 25
- Exit on opposite Donchian breakout or when 1w ADX < 20 (trend weakness)
- Position size fixed at 0.25 to balance risk and return
- Uses 4h primary with 1d HTF for volume and 1w HTF for ADX to filter chop and confirm strong trends
- Target: 75-200 trades over 4 years (19-50/year) by requiring volume spike + strong trend
"""

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
    
    # Donchian(20) channels
    lookback = 20
    upper = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lower = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Get 1d data ONCE before loop for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d volume average (20-period)
    vol_ma_20 = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Get 1w data ONCE before loop for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w ADX(14)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = pd.Series(high_1w - low_1w)
    tr2 = pd.Series(np.abs(high_1w - np.roll(close_1w, 1)))
    tr3 = pd.Series(np.abs(low_1w - np.roll(close_1w, 1)))
    tr_1w = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    dm_plus = pd.Series(np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w),
                                 np.maximum(high_1w - np.roll(high_1w, 1), 0), 0))
    dm_minus = pd.Series(np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)),
                                  np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0))
    
    # Smoothed values
    atr_1w = tr_1w.ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_sm = dm_plus.ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_sm = dm_minus.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_sm / (atr_1w + 1e-10)
    di_minus = 100 * dm_minus_sm / (atr_1w + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx_1w = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align 1d and 1w indicators to 4h timeframe
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Conditions
    volume_spike = volume > (1.5 * vol_ma_20_aligned)  # 1d volume > 1.5x 20-period average
    strong_trend = adx_1w_aligned > 25                 # 1w ADX > 25 indicates strong trend
    weak_trend = adx_1w_aligned < 20                   # 1w ADX < 20 indicates trend weakness
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback, 20, 14, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(volume_spike[i]) if hasattr(volume_spike, '__getitem__') else np.isnan(volume_spike) or
            np.isnan(strong_trend[i]) if hasattr(strong_trend, '__getitem__') else np.isnan(strong_trend) or
            np.isnan(weak_trend[i]) if hasattr(weak_trend, '__getitem__') else np.isnan(weak_trend)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Handle volume_spike, strong_trend, weak_trend as scalars or arrays
        vol_spike = volume_spike[i] if hasattr(volume_spike, '__getitem__') else volume_spike
        str_trend = strong_trend[i] if hasattr(strong_trend, '__getitem__') else strong_trend
        wk_trend = weak_trend[i] if hasattr(weak_trend, '__getitem__') else weak_trend
        
        if position == 0:
            # Long: break above upper band AND volume spike AND strong trend
            if close[i] > upper[i] and vol_spike and str_trend:
                signals[i] = 0.25
                position = 1
            # Short: break below lower band AND volume spike AND strong trend
            elif close[i] < lower[i] and vol_spike and str_trend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: break below lower band OR trend weakness
            if close[i] < lower[i] or wk_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: break above upper band OR trend weakness
            if close[i] > upper[i] or wk_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dVolumeSpike_1wADX_Trend_v1"
timeframe = "4h"
leverage = 1.0