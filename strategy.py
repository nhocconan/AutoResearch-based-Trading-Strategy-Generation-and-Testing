#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with 12h ADX trend filter and volume spike confirmation.
- Uses 6h timeframe (primary) and 12h HTF for ADX trend strength and direction.
- Donchian channels calculated from prior 6h high/low (20-bar lookback).
- Breakout logic: long when price closes above upper Donchian with volume spike and ADX>25,
                  short when price closes below lower Donchian with volume spike and ADX>25.
- Trend filter: only trade when 12h ADX > 25 (strong trend) to avoid choppy markets.
- Volume confirmation: current 6h volume > 1.5 * 20-period 6h volume MA.
- Discrete signal size: 0.25 to balance reward and risk, minimizing fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
- Works in both bull/bear: ADX filter ensures we only trade strong trends, Donchian breakouts capture momentum.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 12h ADX for trend filter (using 12h data)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:  # Need sufficient data for ADX calculation
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate True Range (TR)
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Calculate Directional Movement (+DM, -DM)
    up_move = high_12h[1:] - high_12h[:-1]
    down_move = low_12h[:-1] - low_12h[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (equivalent to EMA with alpha=1/period)
    def WilderSmooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[:period])
        # Subsequent values: Wilder smoothing
        for i in range(period, len(data)):
            if np.isnan(result[i-1]) or np.isnan(data[i]):
                result[i] = np.nan
            else:
                result[i] = result[i-1] - (result[i-1] / period) + (data[i] / period)
        return result
    
    period = 14
    atr_12h = WilderSmooth(tr, period)
    plus_di_12h = 100 * WilderSmooth(plus_dm, period) / atr_12h
    minus_di_12h = 100 * WilderSmooth(minus_dm, period) / atr_12h
    
    # Calculate DX and ADX
    dx = 100 * np.abs(plus_di_12h - minus_di_12h) / (plus_di_12h + minus_di_12h)
    adx_12h = WilderSmooth(dx, period)
    
    # Align 12h ADX to 6h timeframe
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Calculate Donchian channels from prior 6h data (20-bar lookback)
    # Use rolling window on 6h data directly
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_upper = high_roll.values
    donchian_lower = low_roll.values
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    # Trend filter: ADX > 25 indicates strong trend
    strong_trend = adx_12h_aligned > 25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 30)  # Need sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_12h_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price closes above upper Donchian AND strong trend AND volume spike
            if close[i] > donchian_upper[i] and strong_trend[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price closes below lower Donchian AND strong trend AND volume spike
            elif close[i] < donchian_lower[i] and strong_trend[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reverts to middle of Donchian channel or reverse signal
            donchian_middle = (donchian_upper[i] + donchian_lower[i]) / 2
            if close[i] <= donchian_middle:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverts to middle of Donchian channel or reverse signal
            donchian_middle = (donchian_upper[i] + donchian_lower[i]) / 2
            if close[i] >= donchian_middle:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian_20_12hADX_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0