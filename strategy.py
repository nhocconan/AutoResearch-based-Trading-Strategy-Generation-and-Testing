#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1d volume confirmation and 1w ADX trend filter
# Entry: Price breaks above/below 12h Donchian(20) with 1d volume > 1.5x average
# Filter: Only take long when 1w ADX > 25, short when 1w ADX > 25
# Exit: Opposite Donchian break or ADX < 20 (trend weakening)
# Designed to capture strong trends with volume confirmation while avoiding chop
# Target: 15-25 trades/year to minimize fee drag
name = "12h_Donchian_1dVolume_1wADX_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    
    # 1d Volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Get 1w data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ADX (14) on 1w
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = 0  # First value has no previous close
    
    # Directional Movement
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w),
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)),
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    def ma_smoother(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) >= period:
            result[period-1] = np.nansum(arr[:period])
            for i in range(period, len(arr)):
                result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    tr_ma = ma_smoother(tr, 14)
    dm_plus_ma = ma_smoother(dm_plus, 14)
    dm_minus_ma = ma_smoother(dm_minus, 14)
    
    # Directional Indicators
    di_plus = np.where(tr_ma != 0, 100 * dm_plus_ma / tr_ma, 0)
    di_minus = np.where(tr_ma != 0, 100 * dm_minus_ma / tr_ma, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = ma_smoother(dx, 14)
    
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # 12h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Check for NaN values
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(adx_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current 1d volume > 1.5x average
        volume_filter = volume[i] > 1.5 * vol_ma_1d_aligned[i]
        
        # ADX filter: trend strength (ADX > 25)
        strong_trend = adx_1w_aligned[i] > 25
        weak_trend = adx_1w_aligned[i] < 20
        
        if position == 0:
            # Long entry: price breaks above Donchian high + volume + strong trend
            if close[i] > donchian_high[i] and volume_filter and strong_trend:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low + volume + strong trend
            elif close[i] < donchian_low[i] and volume_filter and strong_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below Donchian low OR trend weakens
            if close[i] < donchian_low[i] or weak_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above Donchian high OR trend weakens
            if close[i] > donchian_high[i] or weak_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals