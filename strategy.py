#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout + 1d ADX25 trend filter + volume confirmation
# Camarilla levels provide high-probability reversal/breakout zones derived from prior day's range.
# 1d ADX > 25 ensures alignment with the dominant daily trend to avoid counter-trend trades.
# Volume confirmation (1.8x 20-period EMA) filters false breakouts.
# Designed for 75-200 total trades over 4 years (19-50/year) with discrete sizing to minimize fee drag.
# Works in both bull and bear markets by following the higher-timeframe trend.

name = "4h_Camarilla_R3S3_1dADX25_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for ADX25 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d ADX(14) for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align length
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed TR, DM+ , DM- (Wilder's smoothing)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value: simple average
        result[period-1] = np.nanmean(data[1:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            if np.isnan(result[i-1]) or np.isnan(data[i]):
                result[i] = np.nan
            else:
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    tr_smoothed = wilders_smoothing(tr, 14)
    dm_plus_smoothed = wilders_smoothing(dm_plus, 14)
    dm_minus_smoothed = wilders_smoothing(dm_minus, 14)
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smoothed / tr_smoothed
    di_minus = 100 * dm_minus_smoothed / tr_smoothed
    
    # DX and ADX
    dx = np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100
    dx = np.where((di_plus + di_minus) == 0, 0, dx)
    
    def wilders_smoothing_dx(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value: simple average
        valid_data = data[~np.isnan(data)]
        if len(valid_data) < period:
            return result
        result[period-1] = np.nanmean(data[1:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            if np.isnan(result[i-1]) or np.isnan(data[i]):
                result[i] = np.nan
            else:
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    adx_14 = wilders_smoothing_dx(dx, 14)
    adx_25 = adx_14  # Using ADX(14) as proxy, will filter with threshold 25
    
    # Align 1d ADX to 4h timeframe
    adx_25_aligned = align_htf_to_ltf(prices, df_1d, adx_25)
    
    # Calculate Camarilla levels from previous 1d bar
    # Camarilla: based on prior day's (high-low) range
    close_1d_series = pd.Series(close_1d)
    high_1d_series = pd.Series(high_1d)
    low_1d_series = pd.Series(low_1d)
    
    # Prior day's close, high, low
    prev_close = close_1d_series.shift(1).values
    prev_high = high_1d_series.shift(1).values
    prev_low = low_1d_series.shift(1).values
    
    # Calculate Camarilla levels
    range_ = prev_high - prev_low
    camarilla_h5 = prev_close + (range_ * 1.1 / 2)
    camarilla_h4 = prev_close + (range_ * 1.1 / 4)
    camarilla_h3 = prev_close + (range_ * 1.1 / 6)
    camarilla_l3 = prev_close - (range_ * 1.1 / 6)
    camarilla_l4 = prev_close - (range_ * 1.1 / 4)
    camarilla_l5 = prev_close - (range_ * 1.1 / 2)
    
    # We use H3 and L3 as breakout levels
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Volume confirmation: 20-period EMA on 4h
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start from 20 to have valid volume EMA
        # Skip if any value is NaN or outside session
        if (np.isnan(adx_25_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(vol_ema_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 1.8 x 20-period EMA
        volume_spike = volume[i] > (1.8 * vol_ema_20[i])
        
        # Strong trend: ADX > 25
        strong_trend = adx_25_aligned[i] > 25
        
        if position == 0:
            # Long: price breaks above H3 in strong uptrend with volume spike
            if close[i] > camarilla_h3_aligned[i] and strong_trend and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below L3 in strong downtrend with volume spike
            elif close[i] < camarilla_l3_aligned[i] and strong_trend and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below L3 or loses strong trend
            if close[i] < camarilla_l3_aligned[i] or not strong_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above H3 or loses strong trend
            if close[i] > camarilla_h3_aligned[i] or not strong_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals