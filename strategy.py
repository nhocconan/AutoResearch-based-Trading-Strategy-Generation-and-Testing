#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1w ADX trend filter and volume confirmation
# Long when price breaks above Donchian upper band AND 1w ADX > 25 AND volume > 1.5 * 20-bar average volume
# Short when price breaks below Donchian lower band AND 1w ADX > 25 AND volume > 1.5 * 20-bar average volume
# Exit when price retests the Donchian midpoint (mean of upper and lower bands)
# Uses discrete sizing 0.25 to balance return and fee drag
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# Donchian(20) provides robust price channel structure
# 1w ADX > 25 ensures we only trade in strong trending markets (works in both bull and bear)
# Volume confirmation reduces false breakouts during low participation
# 12h timeframe minimizes fee drag while capturing significant moves

name = "12h_Donchian20_1wADX25_Volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 12h Donchian(20) levels ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate Donchian channels: 20-period high and low
    high_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_20 + low_20) / 2.0
    
    # Align HTF indicators to 12h timeframe (wait for completed 12h bar)
    donchian_20_high_aligned = align_htf_to_ltf(prices, df_12h, high_20)
    donchian_20_low_aligned = align_htf_to_ltf(prices, df_12h, low_20)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_12h, donchian_mid)
    
    # Get 1w data ONCE before loop for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:  # Need enough data for ADX calculation
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w ADX (14-period)
    # True Range
    tr1 = np.abs(high_1w - low_1w)
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = 0  # First period has no previous close
    
    # Directional Movement
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w),
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)),
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+ and DM- using Wilder's smoothing (EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        alpha = 1.0 / period
        # First value is simple average
        if len(data) >= period:
            result[period-1] = np.nanmean(data[:period])
        # Subsequent values
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]):
                result[i] = result[i-1] + alpha * (data[i] - result[i-1])
        return result
    
    tr14 = wilders_smoothing(tr, 14)
    dm_plus_14 = wilders_smoothing(dm_plus, 14)
    dm_minus_14 = wilders_smoothing(dm_minus, 14)
    
    # Directional Indicators
    di_plus = np.where(tr14 != 0, 100 * dm_plus_14 / tr14, 0)
    di_minus = np.where(tr14 != 0, 100 * dm_minus_14 / tr14, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smoothing(dx, 14)
    
    # Align HTF indicators to 12h timeframe (wait for completed 1w bar)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Volume confirmation: volume > 1.5 * 20-bar average volume
    volume_series = pd.Series(volume)
    avg_volume_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirmation = volume > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(donchian_20_high_aligned[i]) or np.isnan(donchian_20_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(adx_1w_aligned[i]) or 
            np.isnan(volume_confirmation[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price > upper band AND strong trend (ADX>25) AND volume confirmation
            if close[i] > donchian_20_high_aligned[i] and adx_1w_aligned[i] > 25 and volume_confirmation[i]:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price < lower band AND strong trend (ADX>25) AND volume confirmation
            elif close[i] < donchian_20_low_aligned[i] and adx_1w_aligned[i] > 25 and volume_confirmation[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price retests midpoint from above
            if close[i] <= donchian_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retests midpoint from below
            if close[i] >= donchian_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals