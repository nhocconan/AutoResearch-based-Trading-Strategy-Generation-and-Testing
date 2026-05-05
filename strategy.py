#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ADX trend filter and volume confirmation
# Long when price breaks above Donchian upper band AND 1d ADX > 25 AND volume > 1.5x 20-period average
# Short when price breaks below Donchian lower band AND 1d ADX > 25 AND volume > 1.5x 20-period average
# Exit when price crosses the Donchian middle band (20-period SMA)
# Uses 4h primary timeframe with 1d HTF for trend filter (ADX) and Donchian structure
# Discrete sizing (0.25) to limit fee drag and manage drawdown
# Target: 75-200 total trades over 4 years (19-50/year) based on proven Donchian breakout performance
# Works in both bull and bear markets by using ADX to filter only strong trends and Donchian for breakouts

name = "4h_Donchian20_Breakout_1dADX25_Trend_Volume"
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
    
    # Get 1d data ONCE before loop for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate ADX on 1d data for trend filter
    # ADX calculation: +DI, -DI, DX, then ADX smoothed
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR, DM+ , DM- (Wilder's smoothing)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(data[1:period])  # skip index 0 (nan)
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    period = 14
    tr_period = wilders_smoothing(tr, period)
    dm_plus_period = wilders_smoothing(dm_plus, period)
    dm_minus_period = wilders_smoothing(dm_minus, period)
    
    # DI+ and DI-
    di_plus = np.where(tr_period != 0, 100 * dm_plus_period / tr_period, 0)
    di_minus = np.where(tr_period != 0, 100 * dm_minus_period / tr_period, 0)
    
    # DX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    
    # ADX (smoothed DX)
    def wilders_smoothing_array(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        result[period-1] = np.nanmean(data[1:period])  # skip index 0 if nan
        for i in range(period, len(data)):
            if np.isnan(result[i-1]):
                result[i] = np.nanmean(data[i-period+1:i+1])
            else:
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    adx_1d = wilders_smoothing_array(dx, period)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Donchian channels on 4h data
    def donchian_channels(high, low, period):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        middle = (upper + lower) / 2
        return upper, lower, middle
    
    donchian_period = 20
    upper_band, lower_band, middle_band = donchian_channels(high, low, donchian_period)
    
    # Volume confirmation: volume > 1.5x 20-period average
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(adx_1d_aligned[i]) or 
            np.isnan(upper_band[i]) or 
            np.isnan(lower_band[i]) or 
            np.isnan(middle_band[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Only trade when ADX > 25 (strong trend)
        if adx_1d_aligned[i] > 25:
            if position == 0:
                # Long conditions: price breaks above upper band AND volume spike
                if close[i] > upper_band[i] and volume_filter[i]:
                    signals[i] = 0.25
                    position = 1
                # Short conditions: price breaks below lower band AND volume spike
                elif close[i] < lower_band[i] and volume_filter[i]:
                    signals[i] = -0.25
                    position = -1
            elif position == 1:
                # Exit long: price crosses below middle band
                if close[i] < middle_band[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price crosses above middle band
                if close[i] > middle_band[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
        else:
            # ADX <= 25: no trend, exit any position
            if position != 0:
                signals[i] = 0.0
                position = 0
    
    return signals