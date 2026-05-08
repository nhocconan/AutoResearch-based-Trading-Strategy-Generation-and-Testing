#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d ADX trend filter and volume confirmation
# Uses Donchian(20) for breakout signals, 1d ADX > 25 for trend strength,
# and volume > 1.5x 20-period average for confirmation.
# Works in bull/bear via breakout logic with trend filter.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

name = "4h_Donchian_1dADX_Volume"
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
    
    # Get daily data for ADX and volume average
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 30:
        return np.zeros(n)
    
    # Calculate daily ADX (14-period)
    # True Range
    tr1 = df_daily['high'].values[1:] - df_daily['low'].values[1:]
    tr2 = np.abs(df_daily['high'].values[1:] - df_daily['close'].values[:-1])
    tr3 = np.abs(df_daily['low'].values[1:] - df_daily['close'].values[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((df_daily['high'].values[1:] - df_daily['high'].values[:-1]) > 
                       (df_daily['low'].values[:-1] - df_daily['low'].values[1:]), 
                       np.maximum(df_daily['high'].values[1:] - df_daily['high'].values[:-1], 0), 0)
    dm_minus = np.where((df_daily['low'].values[:-1] - df_daily['low'].values[1:]) > 
                        (df_daily['high'].values[1:] - df_daily['high'].values[:-1]), 
                        np.maximum(df_daily['low'].values[:-1] - df_daily['low'].values[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Wilder's smoothing
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(data[:period]) / period
        # Subsequent values
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
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
    
    # Calculate daily volume average (20-period EMA)
    daily_volume = df_daily['volume'].values
    vol_ma_20 = pd.Series(daily_volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align daily indicators to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_daily, adx)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_daily, vol_ma_20)
    
    # Calculate 4h Donchian channels (20-period)
    def donchian_channels(high, low, period):
        upper = np.full_like(high, np.nan)
        lower = np.full_like(low, np.nan)
        for i in range(period-1, len(high)):
            upper[i] = np.max(high[i-period+1:i+1])
            lower[i] = np.min(low[i-period+1:i+1])
        return upper, lower
    
    upper, lower = donchian_channels(high, low, 20)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if (np.isnan(adx_aligned[i]) or np.isnan(vol_ma_20_aligned[i]) or 
            np.isnan(upper[i]) or np.isnan(lower[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average
        vol_filter = volume[i] > 1.5 * vol_ma_20_aligned[i]
        
        # ADX filter: > 25 indicates strong trend
        adx_filter = adx_aligned[i] > 25
        
        if position == 0:
            # Look for breakout with volume and trend confirmation
            if close[i] > upper[i] and vol_filter and adx_filter:
                signals[i] = 0.25
                position = 1
            elif close[i] < lower[i] and vol_filter and adx_filter:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price retrace to middle of channel or trend weakens
            mid = (upper[i] + lower[i]) / 2
            if close[i] < mid or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retrace to middle of channel or trend weakens
            mid = (upper[i] + lower[i]) / 2
            if close[i] > mid or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals