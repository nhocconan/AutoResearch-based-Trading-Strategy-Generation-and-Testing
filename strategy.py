#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout + 1d ADX regime filter + volume confirmation
    # Long: Price breaks above Donchian(20) high AND 1d ADX > 25 AND volume > 1.5 * volume MA(20)
    # Short: Price breaks below Donchian(20) low AND 1d ADX > 25 AND volume > 1.5 * volume MA(20)
    # Exit: Price touches Donchian(20) midpoint OR 1d ADX < 20
    # Uses 4h for price action/volume, 1d for ADX regime filter
    # Discrete position sizing (0.25) to minimize fee churn
    # Target: 75-200 total trades over 4 years (~19-50/year) to stay within limits
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian channels and volume (call ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Get 1d data for ADX (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 4h Donchian(20) channels
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    donchian_period = 20
    # Donchian high: rolling max of high
    donchian_high = pd.Series(high_4h).rolling(window=donchian_period, min_periods=donchian_period).max().values
    # Donchian low: rolling min of low
    donchian_low = pd.Series(low_4h).rolling(window=donchian_period, min_periods=donchian_period).min().values
    # Donchian midpoint: average of high and low channels
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Align 4h Donchian to 4h timeframe (no additional delay for price-based indicators)
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_4h, donchian_mid)
    
    # Calculate 4h volume MA(20) for confirmation
    volume_ma = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_4h, volume_ma)
    
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
    
    # Align 1d ADX to 4h (wait for completed 1d bar)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(volume_ma_aligned[i]) or 
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade when 1d ADX > 25 (strong trend)
        strong_trend = adx_aligned[i] > 25
        # Exit regime: ADX < 20 (weak trend/no trend)
        weak_trend = adx_aligned[i] < 20
        
        # Volume confirmation: current volume > 1.5 * 20-period volume MA
        volume_confirmed = volume[i] > 1.5 * volume_ma_aligned[i]
        
        # Donchian breakout signals
        long_breakout = close[i] > donchian_high_aligned[i]
        short_breakout = close[i] < donchian_low_aligned[i]
        
        # Exit when price touches Donchian midpoint
        long_exit = close[i] <= donchian_mid_aligned[i]
        short_exit = close[i] >= donchian_mid_aligned[i]
        
        # Entry logic: Donchian breakout + strong trend regime + volume confirmation
        long_entry = long_breakout and strong_trend and volume_confirmed
        short_entry = short_breakout and strong_trend and volume_confirmed
        
        # Exit logic: Donchian midpoint touch OR regime shifts to weak trend
        long_exit_condition = long_exit or weak_trend
        short_exit_condition = short_exit or weak_trend
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit_condition:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit_condition:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_donchian_breakout_adx_volume_v1"
timeframe = "4h"
leverage = 1.0