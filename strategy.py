#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Bull/Bear Power with 1d ADX regime filter and volume spike.
- Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13 (using 6h EMA13)
- Regime filter: 1d ADX > 25 for trending markets (avoid whipsaws in ranging)
- Volume confirmation: current 6h volume > 1.8 * 20-period 6h volume MA
- Entry: Long when Bull Power > 0 AND Bear Power < prior Bear Power (bullish momentum) AND uptrend regime
         Short when Bear Power < 0 AND Bull Power > prior Bull Power (bearish momentum) AND downtrend regime
- Exit: Reverse signal or power crosses zero
- Discrete signal size: 0.25
- Works in bull/bear: ADX regime filter ensures we only trade strong trends, Elder Ray captures momentum
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 6h EMA13 for Elder Ray
    ema_13_6h = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13_6h  # Bull Power = High - EMA
    bear_power = low - ema_13_6h   # Bear Power = Low - EMA
    
    # Calculate 1d ADX for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smooth TR, DM+, DM- with Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(values, period):
        result = np.zeros_like(values)
        result[period-1] = np.nansum(values[:period])  # First value is simple average
        for i in range(period, len(values)):
            result[i] = result[i-1] - (result[i-1] / period) + values[i]
        return result
    
    period = 14
    tr_period = wilders_smoothing(tr, period)
    dm_plus_period = wilders_smoothing(dm_plus, period)
    dm_minus_period = wilders_smoothing(dm_minus, period)
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_period / tr_period
    di_minus = 100 * dm_minus_period / tr_period
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    # ADX is Wilder's smoothing of DX
    adx_1d = wilders_smoothing(dx, period)
    
    # Align 1d ADX to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Regime filter: ADX > 25 indicates trending market
    trending_regime = adx_1d_aligned > 25
    
    # Volume confirmation: current volume > 1.8 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * volume_ma)
    
    # Previous power values for momentum check
    bull_power_prev = np.roll(bull_power, 1)
    bear_power_prev = np.roll(bear_power, 1)
    bull_power_prev[0] = bull_power[0]  # Initialize first value
    bear_power_prev[0] = bear_power[0]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 13, 14*2)  # Need EMA13, ADX with enough lookback
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power > 0 AND Bull Power rising (momentum) AND trending regime AND volume spike
            if bull_power[i] > 0 and bull_power[i] > bull_power_prev[i] and trending_regime[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 AND Bear Power falling (more negative = momentum) AND trending regime AND volume spike
            elif bear_power[i] < 0 and bear_power[i] < bear_power_prev[i] and trending_regime[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bull Power crosses below zero OR reverse signal
            if bull_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bear Power crosses above zero OR reverse signal
            if bear_power[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_1dADX_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0