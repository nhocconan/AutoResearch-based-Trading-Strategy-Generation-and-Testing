#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ADX(14) regime filter and volume confirmation.
- Long when price breaks above Donchian upper band AND 1d ADX > 25 (trending market) AND volume > 1.5 * median volume
- Short when price breaks below Donchian lower band AND 1d ADX > 25 (trending market) AND volume > 1.5 * median volume
- Exit on opposite Donchian breakout or when 1d ADX < 20 (range market)
- Uses 4h primary timeframe with 1d HTF to target 75-200 total trades over 4 years (19-50/year)
- Donchian channels provide clear breakout levels that work in both bull and bear markets
- 1d ADX ensures we only trade in trending conditions, avoiding whipsaws in ranging markets
- Volume confirmation filters out low-momentum breakouts
- Position size: 0.25 (25% of capital) to manage drawdown in volatile markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period)
    # Upper band = highest high of last 20 bars, Lower band = lowest low of last 20 bars
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Get 1d data ONCE before loop for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d ADX(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = pd.Series(high_1d).rolling(2).max().values - pd.Series(low_1d).rolling(2).min().values
    tr2 = np.abs(pd.Series(high_1d).shift(1).values - pd.Series(close_1d).shift(1).values)
    tr3 = np.abs(pd.Series(low_1d).shift(1).values - pd.Series(close_1d).shift(1).values)
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = high_1d[0] - low_1d[0]  # first value
    
    # Directional Movement
    dm_plus = np.where((pd.Series(high_1d).diff().values > 0) & (pd.Series(high_1d).diff().values > -pd.Series(low_1d).diff().values),
                       pd.Series(high_1d).diff().values, 0.0)
    dm_minus = np.where((-pd.Series(low_1d).diff().values > 0) & (-pd.Series(low_1d).diff().values > pd.Series(high_1d).diff().values),
                        -pd.Series(low_1d).diff().values, 0.0)
    dm_plus[0] = 0.0
    dm_minus[0] = 0.0
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing = EMA with alpha=1/period)
    def WilderSmoothing(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_1d = WilderSmoothing(tr, 14)
    dm_plus_smooth = WilderSmoothing(dm_plus, 14)
    dm_minus_smooth = WilderSmoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr_1d > 0, dm_plus_smooth / atr_1d * 100, 0)
    di_minus = np.where(atr_1d > 0, dm_minus_smooth / atr_1d * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) > 0, np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx_1d = WilderSmoothing(dx, 14)
    
    # Align 1d ADX to 4h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume confirmation: volume > 1.5 * median volume of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    volume_confirm = volume > (1.5 * vol_median)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14*2) + 1  # Donchian(20) + ADX needs ~28 bars for smoothing
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(vol_median[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian upper band, ADX > 25 (strong trend), volume confirmation
            if close[i] > highest_high[i] and adx_1d_aligned[i] > 25 and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower band, ADX > 25 (strong trend), volume confirmation
            elif close[i] < lowest_low[i] and adx_1d_aligned[i] > 25 and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below Donchian lower band OR ADX < 20 (weak trend/ranging)
            if close[i] < lowest_low[i] or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Donchian upper band OR ADX < 20 (weak trend/ranging)
            if close[i] > highest_high[i] or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dADX25_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0