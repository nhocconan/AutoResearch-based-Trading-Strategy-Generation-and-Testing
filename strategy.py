#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Power (Bull/Bear) with 1d ADX regime filter and volume spike confirmation.
- Primary timeframe: 6h for execution, HTF: 1d for Elder Ray and ADX.
- Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13.
- Regime filter: Only trade when 1d ADX > 25 (trending market).
- Volume confirmation: current volume > 2.0x 20-period volume MA.
- Entry: Long when Bull Power > 0 and rising, Short when Bear Power < 0 and falling.
- Exit: When power crosses zero or opposite power expands.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
- Works in bull via buying strength, in bear via selling weakness.
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
    
    # Get 1d data for Elder Ray and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need enough for EMA13 and ADX
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate EMA13 for Elder Ray
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray Power
    bull_power_1d = high_1d - ema_13_1d  # High - EMA13
    bear_power_1d = low_1d - ema_13_1d   # Low - EMA13
    
    # Calculate ADX (14-period)
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
    
    # Smoothed TR, DM+
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align to 6h: use prior 1d's values (completed bar)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 20)  # EMA13 + ADX + volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Only trade in trending market (ADX > 25)
            if adx_aligned[i] > 25:
                # Check for rising Bull Power or falling Bear Power
                if i > 0 and not np.isnan(bull_power_aligned[i-1]) and not np.isnan(bear_power_aligned[i-1]):
                    bull_rising = bull_power_aligned[i] > bull_power_aligned[i-1]
                    bear_falling = bear_power_aligned[i] < bear_power_aligned[i-1]
                    
                    if bull_rising and bull_power_aligned[i] > 0 and volume_spike[i]:
                        # Long on rising Bull Power above zero
                        signals[i] = 0.25
                        position = 1
                    elif bear_falling and bear_power_aligned[i] < 0 and volume_spike[i]:
                        # Short on falling Bear Power below zero
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Long exit: Bull Power crosses zero or Bear Power expands
            if not np.isnan(bull_power_aligned[i]) and not np.isnan(bear_power_aligned[i]):
                if bull_power_aligned[i] <= 0 or bear_power_aligned[i] >= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        elif position == -1:
            # Short exit: Bear Power crosses zero or Bull Power expands
            if not np.isnan(bull_power_aligned[i]) and not np.isnan(bear_power_aligned[i]):
                if bear_power_aligned[i] >= 0 or bull_power_aligned[i] <= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_Power_1dADX_Regime_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0