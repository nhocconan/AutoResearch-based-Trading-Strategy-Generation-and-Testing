#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Bollinger Band mean reversion with 1d ADX regime filter
# In range-bound markets (ADX < 25), price tends to revert to the mean from Bollinger Bands.
# Strong trends (ADX >= 25) are avoided to prevent counter-trend trading.
# Volume confirmation (1.5x 20-period EMA) increases signal reliability.
# Designed for low trade frequency (12-37/year) with discrete sizing to minimize fee drag.
# Works in both bull and bear markets by adapting to regime via ADX filter.

name = "12h_BBMeanReversion_1dADX25_VolumeConfirm"
timeframe = "12h"
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
    
    # Get 1d data for ADX25 regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d ADX(14) for regime filter
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
    adx_regime = adx_14  # Using ADX(14), will filter with threshold 25
    
    # Align 1d ADX to 12h timeframe
    adx_regime_aligned = align_htf_to_ltf(prices, df_1d, adx_regime)
    
    # Bollinger Bands (20, 2) on 12h
    close_series = pd.Series(close)
    ma_20 = close_series.rolling(window=20, min_periods=20).mean()
    std_20 = close_series.rolling(window=20, min_periods=20).std()
    bb_upper = (ma_20 + 2 * std_20).values
    bb_lower = (ma_20 - 2 * std_20).values
    bb_middle = ma_20.values
    
    # Volume confirmation: 20-period EMA on 12h
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start from 20 to have valid BB and volume EMA
        # Skip if any value is NaN or outside session
        if (np.isnan(adx_regime_aligned[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or 
            np.isnan(bb_middle[i]) or np.isnan(vol_ema_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Range-bound market: ADX < 25 (avoid strong trends)
        ranging_market = adx_regime_aligned[i] < 25
        
        # Volume confirmation: current volume > 1.5 x 20-period EMA
        volume_confirm = volume[i] > (1.5 * vol_ema_20[i])
        
        if position == 0:
            # Long: price touches lower BB in ranging market with volume confirmation
            if low[i] <= bb_lower[i] and ranging_market and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price touches upper BB in ranging market with volume confirmation
            elif high[i] >= bb_upper[i] and ranging_market and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price reaches middle BB or exits ranging market
            if close[i] >= bb_middle[i] or not ranging_market:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price reaches middle BB or exits ranging market
            if close[i] <= bb_middle[i] or not ranging_market:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals