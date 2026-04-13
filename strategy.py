#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout with 1d volume spike and 1w ADX regime filter
    # Long: price > Donchian(20) high AND volume > 2.0x 20-period average AND 1w ADX < 25 (range)
    # Short: price < Donchian(20) low AND volume > 2.0x 20-period average AND 1w ADX < 25 (range)
    # Exit: opposite Donchian breakout
    # Using 4h timeframe for optimal trade frequency (target 20-50/year), 1d volume confirmation to avoid false breakouts,
    # and 1w ADX < 25 to trade only in ranging markets where mean reversion works well.
    # Discrete position sizing (0.25) to minimize fee churn.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get 1w data for ADX regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 1d volume 20-period average with min_periods
    vol_ma = np.full(len(df_1d), np.nan)
    for i in range(20, len(df_1d)):
        vol_ma[i] = np.mean(df_1d['volume'].iloc[i-20:i].values)
    
    # Calculate 1w ADX with min_periods
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = np.abs(high_1w[1:] - low_1w[1:])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values
    def smma(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        result[period-1] = np.nansum(arr[1:period])
        for i in range(period, len(arr)):
            if not np.isnan(arr[i]):
                result[i] = result[i-1] - (result[i-1]/period) + arr[i]
        return result
    
    atr = smma(tr, 30)
    dm_plus_smooth = smma(dm_plus, 30)
    dm_minus_smooth = smma(dm_minus, 30)
    
    # DI+ and DI-
    di_plus = np.full_like(atr, np.nan)
    di_minus = np.full_like(atr, np.nan)
    mask = ~np.isnan(atr) & (atr != 0)
    di_plus[mask] = (dm_plus_smooth[mask] / atr[mask]) * 100
    di_minus[mask] = (dm_minus_smooth[mask] / atr[mask]) * 100
    
    # DX and ADX
    dx = np.full_like(atr, np.nan)
    mask_dx = ~np.isnan(di_plus) & ~np.isnan(di_minus) & ((di_plus + di_minus) != 0)
    dx[mask_dx] = (np.abs(di_plus[mask_dx] - di_minus[mask_dx]) / (di_plus[mask_dx] + di_minus[mask_dx])) * 100
    
    adx = smma(dx, 30)
    adx_filter = adx < 25  # Range regime
    
    # Get 4h Donchian(20) for breakout with min_periods
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Align 1d volume spike confirmation
    volume_spike_1d = df_1d['volume'].values > (2.0 * vol_ma)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # Align 1w ADX filter
    adx_filter_aligned = align_htf_to_ltf(prices, df_1w, adx_filter)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_spike_aligned[i]) or np.isnan(adx_filter_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        long_breakout = close[i] > donchian_high[i]
        short_breakout = close[i] < donchian_low[i]
        
        # Entry logic: Breakout + volume confirmation + range regime
        long_entry = long_breakout and volume_spike_aligned[i] and adx_filter_aligned[i]
        short_entry = short_breakout and volume_spike_aligned[i] and adx_filter_aligned[i]
        
        # Exit logic: opposite breakout
        long_exit = short_breakout
        short_exit = long_breakout
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
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

name = "4h_1d_1w_donchian_breakout_volume_adx_filter_v1"
timeframe = "4h"
leverage = 1.0