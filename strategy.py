#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extreme reversal with 1d ADX trend filter and volume spike confirmation
# Long when Williams %R < -80 (oversold) AND 1d ADX > 25 (trending) AND volume > 1.5x 20 EMA
# Short when Williams %R > -20 (overbought) AND 1d ADX > 25 (trending) AND volume > 1.5x 20 EMA
# Williams %R identifies exhaustion points in trending markets; ADX filters for strong trends only
# Volume spike confirms institutional participation at reversal points
# Works in bull markets via buying oversold dips in uptrends and bear markets via selling overbought rallies in downtrends
# Uses discrete sizing (0.25) to limit fee drag. Target: 12-30 trades/year per symbol.

name = "6h_WilliamsR_Extreme_1dADX_VolumeConfirm"
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
    
    # Get 6h data ONCE before loop for Williams %R calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 14:
        return np.zeros(n)
    
    # Calculate 6h Williams %R (14-period)
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    def rolling_max(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window-1, len(arr)):
            result[i] = np.max(arr[i-window+1:i+1])
        return result
    
    def rolling_min(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window-1, len(arr)):
            result[i] = np.min(arr[i-window+1:i+1])
        return result
    
    highest_high = rolling_max(high_6h, 14)
    lowest_low = rolling_min(low_6h, 14)
    williams_r = np.where(
        (highest_high - lowest_low) != 0,
        ((highest_high - close_6h) / (highest_high - lowest_low)) * -100,
        np.nan
    )
    
    # Align 6h Williams %R to prices timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_6h, williams_r)
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough data for ADX calculation
        return np.zeros(n)
    
    # Calculate 1d ADX (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+ and DM- (Wilder's smoothing)
    def WilderSmoothing(arr, period):
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(arr[:period])
        # Subsequent values: prev_smoothed - (prev_smoothed/period) + current_value
        for i in range(period, len(arr)):
            result[i] = result[i-1] - (result[i-1]/period) + arr[i]
        return result
    
    atr = WilderSmoothing(tr, 14)
    dm_plus_smooth = WilderSmoothing(dm_plus, 14)
    dm_minus_smooth = WilderSmoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, (dm_plus_smooth / atr) * 100, 0)
    di_minus = np.where(atr != 0, (dm_minus_smooth / atr) * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 
                  np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    
    def WilderSmoothingDX(arr, period):
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        result[period-1] = np.nansum(arr[period-1:2*period-1]) if 2*period-1 <= len(arr) else np.nan
        for i in range(2*period-1, len(arr)):
            result[i] = result[i-1] - (result[i-1]/period) + arr[i]
        return result
    
    adx = WilderSmoothingDX(dx, 14)
    
    # Align 1d ADX to prices timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Williams %R < -80 (oversold) AND ADX > 25 (trending) AND volume spike
            if (williams_r_aligned[i] < -80 and 
                adx_aligned[i] > 25 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Williams %R > -20 (overbought) AND ADX > 25 (trending) AND volume spike
            elif (williams_r_aligned[i] > -20 and 
                  adx_aligned[i] > 25 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R > -50 (recovering from oversold) OR ADX < 20 (trend weak)
            if (williams_r_aligned[i] > -50 or 
                adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R < -50 (declining from overbought) OR ADX < 20 (trend weak)
            if (williams_r_aligned[i] < -50 or 
                adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals