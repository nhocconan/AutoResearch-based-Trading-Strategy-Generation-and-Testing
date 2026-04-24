#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d volume spike and ADX regime filter.
- Long when close breaks above Donchian(20) high AND 1d ADX > 25 AND volume > 2.0 * 20-period average
- Short when close breaks below Donchian(20) low AND 1d ADX > 25 AND volume > 2.0 * 20-period average
- Exit when price reverses to Donchian(10) midpoint OR ADX < 20 (range market)
- Uses 12h primary with 1d HTF for ADX and volume regime filters to avoid whipsaws
- Donchian channels provide clear structure; ADX filters for trending conditions; volume spike confirms conviction
- Designed to work in both bull (breakouts up) and bear (breakouts down) markets with trend filter
- Signal size: 0.30 discrete levels to balance return and fee drag
- Target: 50-150 total trades over 4 years (12-37/year)
"""

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
    
    # Calculate Donchian channels (20-period)
    def rolling_max(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(window - 1, len(arr)):
            result[i] = np.max(arr[i - window + 1:i + 1])
        return result
    
    def rolling_min(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(window - 1, len(arr)):
            result[i] = np.min(arr[i - window + 1:i + 1])
        return result
    
    donchian_high = rolling_max(high, 20)
    donchian_low = rolling_min(low, 20)
    donchian_mid = (donchian_high + donchian_low) / 2  # 10-period midpoint for exit
    
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
    tr1[0] = tr2[0] = tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    dm_plus = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    dm_minus = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    dm_plus[0] = dm_minus[0] = 0
    
    # Wilder's smoothing
    def wilders_smoothing(values, period):
        result = np.zeros_like(values)
        if len(values) < period:
            return result
        result[period-1] = np.nansum(values[:period])
        for i in range(period, len(values)):
            result[i] = result[i-1] - (result[i-1] / period) + values[i]
        return result
    
    period = 14
    tr_smoothed = wilders_smoothing(tr, period)
    dm_plus_smoothed = wilders_smoothing(dm_plus, period)
    dm_minus_smoothed = wilders_smoothing(dm_minus, period)
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smoothed / np.where(tr_smoothed == 0, 1, tr_smoothed)
    di_minus = 100 * dm_minus_smoothed / np.where(tr_smoothed == 0, 1, tr_smoothed)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / np.where((di_plus + di_minus) == 0, 1, (di_plus + di_minus))
    adx_1d = wilders_smoothing(dx, period)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # 1d volume spike filter
    vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = df_1d['volume'].values > (2.0 * vol_ma_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # Regime filters
    strong_trend = adx_1d_aligned > 25
    weak_trend = adx_1d_aligned < 20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 30) + 5  # Need Donchian20, ADX, and volume data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(vol_spike_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: breakout above Donchian high + strong trend + volume spike
            if close[i] > donchian_high[i] and strong_trend[i] and vol_spike_1d_aligned[i]:
                signals[i] = 0.30
                position = 1
            # Short: breakout below Donchian low + strong trend + volume spike
            elif close[i] < donchian_low[i] and strong_trend[i] and vol_spike_1d_aligned[i]:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Long exit: price returns to midpoint OR weak trend
            if close[i] < donchian_mid[i] or weak_trend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short exit: price returns to midpoint OR weak trend
            if close[i] > donchian_mid[i] or weak_trend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "12h_Donchian20_1dADX_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0