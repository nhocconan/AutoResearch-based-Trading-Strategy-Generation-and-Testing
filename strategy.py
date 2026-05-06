#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 12h ADX trend filter and volume confirmation
# Long when price breaks above Donchian(20) high AND 12h ADX > 25 AND volume > 1.5 * 20-bar average volume
# Short when price breaks below Donchian(20) low AND 12h ADX > 25 AND volume > 1.5 * 20-bar average volume
# Exit when price crosses the 20-bar EMA in opposite direction
# Uses discrete sizing 0.25 to balance return and fee drag
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Donchian provides clear breakout levels, ADX filters for trending markets only, volume confirms participation
# This combination should work in both bull (breakouts continue) and bear (breakdowns continue) markets

name = "6h_Donchian20_12hADX25_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 6h Donchian(20) channels
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate 6h 20-bar EMA for exit
    close_series = pd.Series(close)
    ema20 = close_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume confirmation: volume > 1.5 * 20-bar average volume
    volume_series = pd.Series(volume)
    avg_volume_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirmation = volume > (1.5 * avg_volume_20)
    
    # Get 12h data ONCE before loop for ADX trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h ADX(14)
    # True Range
    tr1 = np.abs(high_12h[1:] - low_12h[1:])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Directional Movement
    dm_plus = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    dm_minus = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed TR, DM+ and DM- using Wilder's smoothing (equivalent to EMA with alpha=1/period)
    def wilders_smoothing(values, period):
        result = np.full_like(values, np.nan)
        if len(values) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(values[:period])
        # Subsequent values: smoothed = prev_smoothed - (prev_smoothed/period) + current_value
        for i in range(period, len(values)):
            if not np.isnan(values[i]):
                result[i] = result[i-1] - (result[i-1]/period) + values[i]
        return result
    
    atr = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smoothing(dx, 14)
    
    # Align HTF indicators to 6h timeframe (wait for completed HTF bar)
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema20[i]) or np.isnan(adx_aligned[i]) or np.isnan(volume_confirmation[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price > Donchian high AND uptrend (ADX > 25) AND volume confirmation
            if close[i] > donchian_high[i] and adx_aligned[i] > 25 and volume_confirmation[i]:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price < Donchian low AND downtrend (ADX > 25) AND volume confirmation
            elif close[i] < donchian_low[i] and adx_aligned[i] > 25 and volume_confirmation[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below 20-bar EMA
            if close[i] < ema20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above 20-bar EMA
            if close[i] > ema20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals