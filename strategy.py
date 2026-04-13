#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 12h Donchian breakout + 1d ADX regime filter
    # Long: price breaks above Donchian(20) high AND 1d ADX > 25 (strong trend)
    # Short: price breaks below Donchian(20) low AND 1d ADX > 25 (strong trend)
    # Exit: price closes opposite Donchian band OR 1d ADX < 20 (weak trend)
    # Uses 12h for price action/DN, 1d for ADX regime
    # Discrete position sizing (0.30) to balance return and drawdown
    # Target: 50-150 total trades over 4 years (~12-37/year) to stay within limits
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 12h data for Donchian channels (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Get 1d data for ADX (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 12h Donchian channels (20-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Rolling max/min for Donchian channels
    def rolling_max(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            result[i] = np.max(arr[i-window+1:i+1])
        return result
    
    def rolling_min(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            result[i] = np.min(arr[i-window+1:i+1])
        return result
    
    dn_high_20 = rolling_max(high_12h, 20)
    dn_low_20 = rolling_min(low_12h, 20)
    
    # Align 12h Donchian to 12h timeframe (no additional delay for price-based indicators)
    dn_high_aligned = align_htf_to_ltf(prices, df_12h, dn_high_20)
    dn_low_aligned = align_htf_to_ltf(prices, df_12h, dn_low_20)
    
    # Calculate 1d ADX (Average Directional Index)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Wilder's smoothing for TR, DM+, DM-
    def wilders_smoothing(data, period):
        alpha = 1.0 / period
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nanmean(data[:period])
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_1d = wilders_smoothing(tr, 14)
    dm_plus_smoothed = wilders_smoothing(dm_plus, 14)
    dm_minus_smoothed = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.full_like(atr_1d, np.nan)
    di_minus = np.full_like(atr_1d, np.nan)
    mask = ~np.isnan(atr_1d) & (atr_1d > 0)
    di_plus[mask] = 100 * dm_plus_smoothed[mask] / atr_1d[mask]
    di_minus[mask] = 100 * dm_minus_smoothed[mask] / atr_1d[mask]
    
    # DX and ADX
    dx = np.full_like(atr_1d, np.nan)
    dx_mask = mask & ~np.isnan(di_plus) & ~np.isnan(di_minus) & ((di_plus + di_minus) > 0)
    dx[dx_mask] = 100 * np.abs(di_plus[dx_mask] - di_minus[dx_mask]) / (di_plus[dx_mask] + di_minus[dx_mask])
    
    adx_1d = wilders_smoothing(dx, 14)
    
    # Align 1d ADX to 12h (wait for completed 1d bar)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(dn_high_aligned[i]) or np.isnan(dn_low_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade when 1d ADX > 25 (strong trend)
        strong_trend = adx_aligned[i] > 25
        # Exit regime: ADX < 20 (weak trend/no trend)
        weak_trend = adx_aligned[i] < 20
        
        # Donchian breakout signals
        long_breakout = close[i] > dn_high_aligned[i]
        short_breakout = close[i] < dn_low_aligned[i]
        
        # Exit conditions: close opposite band OR regime shift
        long_exit = close[i] < dn_low_aligned[i]
        short_exit = close[i] > dn_high_aligned[i]
        regime_exit = weak_trend
        
        # Entry logic: Donchian breakout + strong trend regime
        long_entry = long_breakout and strong_trend
        short_entry = short_breakout and strong_trend
        
        # Exit logic: opposite breakout OR regime shifts to weak trend
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.30
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.30
        elif position == 1 and (long_exit or regime_exit):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (short_exit or regime_exit):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.30
            elif position == -1:
                signals[i] = -0.30
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_donchian_breakout_adx_regime_v1"
timeframe = "12h"
leverage = 1.0