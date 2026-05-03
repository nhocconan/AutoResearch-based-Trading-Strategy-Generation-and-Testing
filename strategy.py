#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w volume confirmation and ADX trend filter
# Donchian breakout captures sustained momentum in both bull and bear markets.
# Weekly volume spike confirms institutional participation.
# ADX > 25 ensures we only trade in trending conditions, avoiding choppy markets.
# Designed for low trade frequency (target: 7-25 trades/year) to minimize fee drag.
# Works in bull markets via breakouts and in bear markets via breakdowns.

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
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR, +DM, -DM using Wilder's smoothing (EMA with alpha=1/14)
    atr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Plus and Minus Directional Indicators
    di_plus = 100 * dm_plus_14 / np.where(atr_14 == 0, 1, atr_14)
    di_minus = 100 * dm_minus_14 / np.where(atr_14 == 0, 1, atr_14)
    
    # Directional Index (DX) and ADX
    dx = 100 * np.abs(di_plus - di_minus) / np.where((di_plus + di_minus) == 0, 1, (di_plus + di_minus))
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align 1w indicators to 1d timeframe
    volume_spike_aligned = align_htf_to_ltf(prices, df_1w, volume_spike)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Calculate 1d Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_spike_aligned[i]) or np.isnan(adx_aligned[i]) or 
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: ADX > 25 indicates trending market
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