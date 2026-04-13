#!/usr/bin/env python3
"""
12h_1d_Adaptive_Range_Breakout_with_Volume_and_ADX
Hypothesis: In ranging markets (ADX < 25), price tends to revert from Bollinger Bands (20,2) with volume confirmation.
In trending markets (ADX > 25), price breaks Donchian channels (20) with volume confirmation.
Uses 1-day ADX for regime detection to avoid whipsaws. Works in both bull (trend breaks) and bear (mean reversion in ranges).
Target: 15-30 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for regime filter (ADX) and trend filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX(14) on daily data
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = np.abs(high[1:] - low[1:])
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        
        # Directional Movement
        dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
        dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
        dm_plus = np.concatenate([[np.nan], dm_plus])
        dm_minus = np.concatenate([[np.nan], dm_minus])
        
        # Smooth TR, DM+ and DM- with Wilder's smoothing
        def wilder_smooth(data, period):
            result = np.full_like(data, np.nan)
            if len(data) < period:
                return result
            # First value: simple average
            result[period-1] = np.nansum(data[1:period]) if np.any(np.isnan(data[1:period])) else np.sum(data[1:period]) / period
            # Subsequent values
            for i in range(period, len(data)):
                if np.isnan(result[i-1]) or np.isnan(data[i]):
                    result[i] = np.nan
                else:
                    result[i] = result[i-1] - (result[i-1] / period) + data[i]
            return result
        
        tr_smooth = wilder_smooth(tr, period)
        dm_plus_smooth = wilder_smooth(dm_plus, period)
        dm_minus_smooth = wilder_smooth(dm_minus, period)
        
        # Directional Indicators
        di_plus = 100 * dm_plus_smooth / tr_smooth
        di_minus = 100 * dm_minus_smooth / tr_smooth
        
        # Avoid division by zero
        di_sum = di_plus + di_minus
        dx = np.where(di_sum != 0, 100 * np.abs(di_plus - di_minus) / di_sum, 0)
        
        # ADX: smoothed DX
        adx = wilder_smooth(dx, period)
        return adx
    
    adx_14 = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # Bollinger Bands (20,2) for mean reversion
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean()
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std()
    bb_upper = sma_20 + 2 * std_20
    bb_lower = sma_20 - 2 * std_20
    
    # Donchian Channel (20) for breakouts
    def donchian_channel(high, low, window=20):
        upper = pd.Series(high).rolling(window=window, min_periods=window).max()
        lower = pd.Series(low).rolling(window=window, min_periods=window).min()
        return upper.values, lower.values
    
    dc_upper, dc_lower = donchian_channel(high, low, 20)
    
    # Volume confirmation: current volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 1.3)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(adx_14_aligned[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or
            np.isnan(dc_upper[i]) or np.isnan(dc_lower[i]) or np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        adx = adx_14_aligned[i]
        is_ranging = adx < 25  # Regime filter: ranging market
        is_trending = adx >= 25  # Trending market
        
        if is_ranging:
            # Mean reversion: fade Bollinger Bands with volume
            long_condition = (close[i] <= bb_lower[i]) and volume_expansion[i]
            short_condition = (close[i] >= bb_upper[i]) and volume_expansion[i]
        else:
            # Trend following: break Donchian channels with volume
            long_condition = (close[i] > dc_upper[i]) and volume_expansion[i]
            short_condition = (close[i] < dc_lower[i]) and volume_expansion[i]
        
        if long_condition and position != 1:
            position = 1
            signals[i] = position_size
        elif short_condition and position != -1:
            position = -1
            signals[i] = -position_size
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "12h_1d_Adaptive_Range_Breakout_with_Volume_and_ADX"
timeframe = "12h"
leverage = 1.0