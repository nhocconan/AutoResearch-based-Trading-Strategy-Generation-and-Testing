#!/usr/bin/env python3
"""
4h ADX + Donchian Breakout + Volume Spike Strategy
Trends in 4h timeframe with ADX filter to avoid chop, Donchian breakout for entry,
and volume spike for confirmation. Designed for low trade frequency with clear trend-following edge.
Works in both bull and bear markets by filtering for strong trends only.
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
    
    # Get daily data for ADX trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate ADX(14) on daily timeframe
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values
    def _wilder_smooth(arr, period):
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(arr[1:period])
        # Subsequent values: Wilder smoothing
        for i in range(period, len(arr)):
            if not np.isnan(result[i-1]) and not np.isnan(arr[i]):
                result[i] = result[i-1] - (result[i-1] / period) + arr[i]
            else:
                result[i] = np.nan
        return result
    
    atr = _wilder_smooth(tr, 14)
    di_plus = _wilder_smooth(dm_plus, 14)
    di_minus = _wilder_smooth(dm_minus, 14)
    
    # DI values
    di_plus_val = np.where(atr != 0, di_plus / atr * 100, 0)
    di_minus_val = np.where(atr != 0, di_minus / atr * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus_val + di_minus_val) != 0, 
                  np.abs(di_plus_val - di_minus_val) / (di_plus_val + di_minus_val) * 100, 0)
    adx = _wilder_smooth(dx, 14)
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Donchian channels (20-period) on 4h
    def donchian_channels(high, low, period):
        upper = np.full_like(high, np.nan)
        lower = np.full_like(low, np.nan)
        for i in range(period-1, len(high)):
            upper[i] = np.max(high[i-period+1:i+1])
            lower[i] = np.min(low[i-period+1:i+1])
        return upper, lower
    
    donch_upper, donch_lower = donchian_channels(high, low, 20)
    
    # Volume spike (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50  # need enough history for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # ADX threshold for trend strength (>25 indicates strong trend)
        strong_trend = adx_aligned[i] > 25
        
        price = close[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper + strong trend + volume spike
            if price > donch_upper[i] and strong_trend and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower + strong trend + volume spike
            elif price < donch_lower[i] and strong_trend and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position management
            signals[i] = 0.25
            # Exit: price breaks below Donchian lower or trend weakens
            if price < donch_lower[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position management
            signals[i] = -0.25
            # Exit: price breaks above Donchian upper or trend weakens
            if price > donch_upper[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_ADX_Donchian_Breakout_VolumeSpike"
timeframe = "4h"
leverage = 1.0