#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w volume spike + ADX trend filter
# Donchian breakout captures sustained momentum, 1w volume spike confirms institutional interest,
# ADX > 25 ensures we only trade in clear trending markets to avoid whipsaws in ranging conditions.
# Designed to work in both bull and bear markets by trading with the trend.
# Target: 7-25 trades/year (30-100 over 4 years).

name = "1d_Donchian20_1wVolumeSpike_ADXTrend"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1w data for volume spike and ADX
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 1w volume spike (volume > 2.0 * 20-period EMA of volume)
    vol_ema_20 = pd.Series(df_1w['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = df_1w['volume'].values > (2.0 * vol_ema_20)
    
    # Calculate 1w ADX(14)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = np.abs(high_1w[1:] - low_1w[:-1])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Plus Directional Movement (+DM) and Minus Directional Movement (-DM)
    dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    
    # Smoothed TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(data[:period])
        # Subsequent values: smoothed = previous_smoothed - (previous_smoothed / period) + current_value
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr_1w = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / np.where(atr_1w == 0, np.nan, atr_1w)
    di_minus = 100 * dm_minus_smooth / np.where(atr_1w == 0, np.nan, atr_1w)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / np.where((di_plus + di_minus) == 0, np.nan, (di_plus + di_minus))
    adx = wilders_smoothing(dx, 14)
    
    # Align 1w indicators to 1d timeframe
    volume_spike_aligned = align_htf_to_ltf(prices, df_1w, volume_spike)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Calculate 1d Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Increased warmup for ADX calculation
        # Skip if any value is NaN or outside session
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_spike_aligned[i]) or np.isnan(adx_aligned[i]) or 
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: ADX > 25 indicates strong trend
        is_trending = adx_aligned[i] > 25
        
        if position == 0:
            # Long: Donchian breakout above upper band + volume spike + trending
            if high[i] > highest_high[i] and volume_spike_aligned[i] and is_trending:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakdown below lower band + volume spike + trending
            elif low[i] < lowest_low[i] and volume_spike_aligned[i] and is_trending:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Donchian breakdown below middle band OR reverse signal
            middle = (highest_high[i] + lowest_low[i]) / 2
            if low[i] < middle or (low[i] < lowest_low[i] and volume_spike_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Donchian breakout above middle band OR reverse signal
            middle = (highest_high[i] + lowest_low[i]) / 2
            if high[i] > middle or (high[i] > highest_high[i] and volume_spike_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals