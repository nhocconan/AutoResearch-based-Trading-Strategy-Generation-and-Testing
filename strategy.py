#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h Donchian(20) Breakout + Volume + 1d ADX Trend Filter
# Hypothesis: Trade breakouts of 20-period Donchian channels on 12h timeframe
# with volume confirmation and 1d ADX > 25 to ensure trending markets.
# Works in both bull and bear by only taking breakouts in the direction of
# the daily trend (ADX > 25 + DI+ > DI- for long, DI- > DI+ for short).
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

name = "12h_donchian20_volume_1d_adx_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for ADX trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 30:
        return np.zeros(n)
    
    # Calculate 12h Donchian channels (20-period)
    high_12h = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_12h = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate daily ADX (14-period)
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # True Range
    tr1 = high_daily - low_daily
    tr2 = np.abs(high_daily - np.roll(close_daily, 1))
    tr3 = np.abs(low_daily - np.roll(close_daily, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((high_daily - np.roll(high_daily, 1)) > (np.roll(low_daily, 1) - low_daily),
                       np.maximum(high_daily - np.roll(high_daily, 1), 0), 0)
    dm_minus = np.where((np.roll(low_daily, 1) - low_daily) > (high_daily - np.roll(high_daily, 1)),
                        np.maximum(np.roll(low_daily, 1) - low_daily, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    def smooth_wilder(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(arr[1:period]) if period > 1 else 0
        # Wilder smoothing
        for i in range(period, len(arr)):
            result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    tr_smooth = smooth_wilder(tr, 14)
    dm_plus_smooth = smooth_wilder(dm_plus, 14)
    dm_minus_smooth = smooth_wilder(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(tr_smooth != 0, 100 * dm_plus_smooth / tr_smooth, 0)
    di_minus = np.where(tr_smooth != 0, 100 * dm_minus_smooth / tr_smooth, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = smooth_wilder(dx, 14)
    
    # Align daily ADX and DI to 12h
    adx_12h = align_htf_to_ltf(prices, df_daily, adx)
    di_plus_12h = align_htf_to_ltf(prices, df_daily, di_plus)
    di_minus_12h = align_htf_to_ltf(prices, df_daily, di_minus)
    
    # Volume filter: 12h volume > 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(high_12h[i]) or np.isnan(low_12h[i]) or 
            np.isnan(adx_12h[i]) or np.isnan(di_plus_12h[i]) or np.isnan(di_minus_12h[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_ok = volume[i] > vol_ma_20[i]
        
        # Trend filter: ADX > 25 and DI+ > DI- for uptrend, DI- > DI+ for downtrend
        uptrend = adx_12h[i] > 25 and di_plus_12h[i] > di_minus_12h[i]
        downtrend = adx_12h[i] > 25 and di_minus_12h[i] > di_plus_12h[i]
        
        if position == 1:  # Long position
            # Exit: price reaches 12h lower Donchian or trend changes
            if low[i] <= low_12h[i] or not uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price reaches 12h upper Donchian or trend changes
            if high[i] >= high_12h[i] or not downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Breakout in direction of daily trend with volume
            if vol_ok:
                if uptrend and high[i] > high_12h[i]:  # Upward breakout in uptrend
                    position = 1
                    signals[i] = 0.25
                elif downtrend and low[i] < low_12h[i]:  # Downward breakout in downtrend
                    position = -1
                    signals[i] = -0.25
    
    return signals