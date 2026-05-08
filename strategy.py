#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R with 1d ADX trend and volume confirmation
# Uses 4h Williams %R for mean reversion entries, filtered by daily ADX (>25) for trend strength
# and volume spike (>2x 20-period average). Designed to capture reversals in both bull and bear
# markets with high win rate. Target: 20-50 trades/year (80-200 total over 4 years).

name = "4h_WilliamsR_1dADX_Trend_VolumeSpike"
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
    
    # Get daily data for ADX
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 14:
        return np.zeros(n)
    
    # Calculate daily ADX(14)
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # True Range
    tr1 = high_daily[1:] - low_daily[1:]
    tr2 = np.abs(high_daily[1:] - close_daily[:-1])
    tr3 = np.abs(low_daily[1:] - close_daily[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_daily[1:] - high_daily[:-1]) > (low_daily[:-1] - low_daily[1:]),
                       np.maximum(high_daily[1:] - high_daily[:-1], 0), 0)
    dm_minus = np.where((low_daily[:-1] - low_daily[1:]) > (high_daily[1:] - high_daily[:-1]),
                        np.maximum(low_daily[:-1] - low_daily[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values
    def smooth_wilder(arr, period):
        smoothed = np.full_like(arr, np.nan)
        if len(arr) < period:
            return smoothed
        smoothed[period-1] = np.nansum(arr[1:period])
        for i in range(period, len(arr)):
            if np.isnan(smoothed[i-1]):
                smoothed[i] = np.nansum(arr[i-period+1:i+1])
            else:
                smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + arr[i]
        return smoothed
    
    atr = smooth_wilder(tr, 14)
    dm_plus_smooth = smooth_wilder(dm_plus, 14)
    dm_minus_smooth = smooth_wilder(dm_minus, 14)
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = np.full_like(dx, np.nan)
    if len(dx) >= 14:
        adx[13] = np.nanmean(dx[1:14])
        for i in range(14, len(dx)):
            adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # Calculate 4h Williams %R(14)
    def williams_r(high_arr, low_arr, close_arr, period):
        wr = np.full_like(close_arr, np.nan)
        for i in range(period-1, len(close_arr)):
            highest_high = np.max(high_arr[i-period+1:i+1])
            lowest_low = np.min(low_arr[i-period+1:i+1])
            if highest_high - lowest_low != 0:
                wr[i] = -100 * (highest_high - close_arr[i]) / (highest_high - lowest_low)
        return wr
    
    wr14 = williams_r(high, low, close, 14)
    
    # Calculate 4h volume average for volume spike
    vol_avg_20 = np.full(n, np.nan)
    if n >= 20:
        for i in range(20, n):
            vol_avg_20[i] = np.mean(volume[i-20:i])
    
    # Align daily ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_daily, adx)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(14, 20)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if np.isnan(wr14[i]) or np.isnan(adx_aligned[i]) or np.isnan(vol_avg_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Check conditions
        strong_trend = adx_aligned[i] > 25
        vol_spike = volume[i] > 2.0 * vol_avg_20[i]
        
        if position == 0:
            # Look for entry: Williams %R oversold/overbought with trend and volume confirmation
            if wr14[i] < -80 and strong_trend and vol_spike:
                signals[i] = 0.25
                position = 1
            elif wr14[i] > -20 and strong_trend and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R overbought or trend weakens or volume drops
            if wr14[i] > -20 or not strong_trend or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R oversold or trend weakens or volume drops
            if wr14[i] < -80 or not strong_trend or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals