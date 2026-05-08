#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d ADX trend filter + volume confirmation
# Long when price breaks above Donchian high (20) and 1d ADX > 25 (trending) and volume > 1.5x average
# Short when price breaks below Donchian low (20) and 1d ADX > 25 and volume > 1.5x average
# Exit when price returns to Donchian middle (10-period average of high/low) or ADX < 20
# Donchian breakouts capture strong moves, ADX filters for trending markets only, volume confirms strength
# Targets 150-250 total trades over 4 years (38-63/year) for optimal balance of edge and frequency

name = "4h_Donchian20_1dADX_Trend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    def highest(high, length):
        result = np.full_like(high, np.nan, dtype=float)
        for i in range(length-1, len(high)):
            result[i] = np.max(high[i-length+1:i+1])
        return result
    
    def lowest(low, length):
        result = np.full_like(low, np.nan, dtype=float)
        for i in range(length-1, len(low)):
            result[i] = np.min(low[i-length+1:i+1])
        return result
    
    donchian_high = highest(high, 20)
    donchian_low = lowest(low, 20)
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ADX (14-period) on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align with indices
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed TR, DM+, DM- (using Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smoothing(arr, period):
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(arr[1:period]) if np.any(~np.isnan(arr[1:period])) else np.nan
        # Wilder's smoothing: new_value = (prev * (period-1) + current) / period
        for i in range(period, len(arr)):
            if not np.isnan(result[i-1]) and not np.isnan(arr[i]):
                result[i] = (result[i-1] * (period-1) + arr[i]) / period
            else:
                result[i] = np.nan
        return result
    
    tr_smoothed = wilders_smoothing(tr, 14)
    dm_plus_smoothed = wilders_smoothing(dm_plus, 14)
    dm_minus_smoothed = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(tr_smoothed != 0, (dm_plus_smoothed / tr_smoothed) * 100, 0)
    di_minus = np.where(tr_smoothed != 0, (dm_minus_smoothed / tr_smoothed) * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 
                  np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx = wilders_smoothing(dx, 14)
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_conf = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        donchian_high_val = donchian_high[i]
        donchian_low_val = donchian_low[i]
        donchian_mid_val = donchian_mid[i]
        adx_val = adx_aligned[i]
        vol_conf_val = vol_conf[i]
        
        if position == 0:
            # Enter long: price breaks above Donchian high, ADX > 25 (trending), volume confirmation
            if close_val > donchian_high_val and adx_val > 25 and vol_conf_val:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low, ADX > 25 (trending), volume confirmation
            elif close_val < donchian_low_val and adx_val > 25 and vol_conf_val:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Donchian mid or ADX < 20 (trend weakening)
            if close_val < donchian_mid_val or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Donchian mid or ADX < 20 (trend weakening)
            if close_val > donchian_mid_val or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals