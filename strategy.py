#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Index with 1d ADX regime filter and volume spike confirmation.
- Primary timeframe: 6h targeting 80-150 total trades over 4 years (20-38/year).
- HTF: 1d for ADX trend strength and Elder Ray calculation.
- Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13.
- Entry: Long when Bull Power > 0 AND ADX > 25 AND volume > 1.5x 20-period average volume.
         Short when Bear Power < 0 AND ADX > 25 AND volume > 1.5x 20-period average volume.
- Exit: Opposite Elder Ray signal OR ADX < 20 (trend weakening).
- Signal size: 0.25 discrete to minimize fee drag.
- ADX > 25 ensures we only trade in trending markets, avoiding whipsaws in ranges.
- Volume confirmation ensures breakouts have participation.
- Works in bull markets (buy strength in uptrend) and bear markets (sell weakness in downtrend).
- Estimated trades: ~100 total over 4 years (~25/year) based on 6h timeframe with strict filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def adx(high, low, close, period):
    """Calculate Average Directional Index."""
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr_period = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
    dm_plus_period = pd.Series(dm_plus).ewm(span=period, adjust=False, min_periods=period).mean().values
    dm_minus_period = pd.Series(dm_minus).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_period / (tr_period + 1e-10)
    di_minus = 100 * dm_minus_period / (tr_period + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx_values = pd.Series(dx).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    return adx_values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # EMA13 for Elder Ray calculation
    ema13_1d = ema(df_1d['close'].values, 13)
    
    # Bull Power and Bear Power
    bull_power_1d = df_1d['high'].values - ema13_1d
    bear_power_1d = df_1d['low'].values - ema13_1d
    
    # ADX for trend strength
    adx_1d = adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    
    # Align to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume spike filter (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma_20 + 1e-10)  # Avoid division by zero
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 40  # Need sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Exit conditions: opposite Elder Ray signal OR ADX < 20 (trend weakening)
        if position != 0:
            # Exit long: Bear Power >= 0 OR ADX < 20
            if position == 1:
                if bear_power_aligned[i] >= 0 or adx_aligned[i] < 20:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: Bull Power <= 0 OR ADX < 20
            elif position == -1:
                if bull_power_aligned[i] <= 0 or adx_aligned[i] < 20:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Elder Ray with trend strength and volume confirmation
        if position == 0:
            # Long: Bull Power > 0 AND ADX > 25 AND volume spike
            if bull_power_aligned[i] > 0 and adx_aligned[i] > 25 and vol_ratio[i] > 1.5:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 AND ADX > 25 AND volume spike
            elif bear_power_aligned[i] < 0 and adx_aligned[i] > 25 and vol_ratio[i] > 1.5:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_1dADX_RegimeFilter_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0