#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extreme + 12h volume spike + 1d ADX trend filter
# Long when Williams %R(14) < -80 (oversold) AND 12h volume > 2.0x 24-period average AND 1d ADX > 25 (trending)
# Short when Williams %R(14) > -20 (overbought) AND 12h volume > 2.0x 24-period average AND 1d ADX > 25 (trending)
# Exit when Williams %R crosses above -50 (for long) or below -50 (for short) OR ADX < 20 (trend weak)
# Uses discrete sizing (0.25) to limit fee drag. Target: 15-25 trades/year per symbol.
# Williams %R identifies exhaustion points, volume spike confirms institutional interest at extremes,
# 1d ADX ensures we only trade in trending environments to avoid chop whipsaws.
# Works in bull markets via buying oversold dips in uptrends and bear markets via selling overbought rallies in downtrends.

name = "6h_WilliamsR_EXTREME_12hVolumeSpike_1dADX_Trend"
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
    
    # Calculate Williams %R on 6h data (14-period)
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    highest_high = pd.Series(high_6h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_6h).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_6h) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align Williams %R to prices timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_6h, williams_r)
    
    # Get 12h data for volume spike filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 24:
        return np.zeros(n)
    
    # Calculate 12h volume spike: volume > 2.0x 24-period average
    volume_12h = df_12h['volume'].values
    vol_ma_24 = pd.Series(volume_12h).rolling(window=24, min_periods=24).mean().values
    volume_spike_12h = volume_12h > (2.0 * vol_ma_24)
    
    # Align 12h volume spike to prices timeframe
    volume_spike_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_spike_12h.astype(float))
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    # Calculate 1d ADX (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smooth TR, DM+, DM- using Wilder's smoothing (equivalent to EMA with alpha=1/14)
    def wilder_smooth(data, period):
        """Wilder's smoothing: first value is SMA, then recursive"""
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value: simple average
        result[period-1] = np.nanmean(data[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
            else:
                result[i] = np.nan
        return result
    
    tr_smooth = wilder_smooth(tr, 14)
    dm_plus_smooth = wilder_smooth(dm_plus, 14)
    dm_minus_smooth = wilder_smooth(dm_minus, 14)
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / tr_smooth
    di_minus = 100 * dm_minus_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    # Handle division by zero
    dx = np.where((di_plus + di_minus) == 0, 0, dx)
    
    adx = wilder_smooth(dx, 14)
    
    # ADX > 25 indicates strong trend
    adx_strong = adx > 25
    adx_weak = adx < 20  # For exit condition
    
    # Align 1d ADX to prices timeframe
    adx_strong_aligned = align_htf_to_ltf(prices, df_1d, adx_strong.astype(float))
    adx_weak_aligned = align_htf_to_ltf(prices, df_1d, adx_weak.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(volume_spike_12h_aligned[i]) or 
            np.isnan(adx_strong_aligned[i]) or 
            np.isnan(adx_weak_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Williams %R < -80 (oversold) AND volume spike AND strong trend (ADX > 25)
            if (williams_r_aligned[i] < -80 and 
                volume_spike_12h_aligned[i] > 0.5 and 
                adx_strong_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short conditions: Williams %R > -20 (overbought) AND volume spike AND strong trend (ADX > 25)
            elif (williams_r_aligned[i] > -20 and 
                  volume_spike_12h_aligned[i] > 0.5 and 
                  adx_strong_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R crosses above -50 OR trend weakens (ADX < 20)
            if (williams_r_aligned[i] > -50 or 
                adx_weak_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses below -50 OR trend weakens (ADX < 20)
            if (williams_r_aligned[i] < -50 or 
                adx_weak_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals