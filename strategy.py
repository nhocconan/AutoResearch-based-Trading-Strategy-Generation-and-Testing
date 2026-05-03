#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume confirmation
# Uses 4h EMA50 for trend direction (works in bull/bear via alignment) and 1d ADX > 25 for regime filter
# Camarilla levels provide precise intraday breakout points with volume confirmation to avoid false signals
# Designed for 60-150 total trades over 4 years (15-37/year) on 1h timeframe with discrete sizing

name = "1h_Camarilla_R3S3_4hEMA50_1dADX25_VolumeSpike"
timeframe = "1h"
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
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data for ADX25 regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d ADX(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Wilder's smoothing function
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
    adx_25_aligned = align_htf_to_ltf(prices, df_1d, adx_14)  # Using ADX(14), filter with >25
    
    # Calculate Camarilla levels from previous 1h bar
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.25*(high-low), etc.
    # We use R3/S3 for breakout entries
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    
    # Previous bar's high, low, close
    prev_high = high_series.shift(1)
    prev_low = low_series.shift(1)
    prev_close = close_series.shift(1)
    
    # Camarilla levels
    camarilla_range = prev_high - prev_low
    r3 = prev_close + 1.25 * camarilla_range
    s3 = prev_close - 1.25 * camarilla_range
    r3_aligned = r3.values
    s3_aligned = s3.values
    
    # Volume confirmation: 20-period EMA on 1h
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start from 20 to have valid volume EMA and Camarilla
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(adx_25_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(vol_ema_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 2.0 x 20-period EMA
        volume_spike = volume[i] > (2.0 * vol_ema_20[i])
        
        # Strong trend regime: ADX > 25
        strong_regime = adx_25_aligned[i] > 25
        
        # 4h trend: price above/below EMA50
        uptrend_4h = close[i] > ema_50_4h_aligned[i]
        downtrend_4h = close[i] < ema_50_4h_aligned[i]
        
        if position == 0:
            # Long: price breaks above R3 in 4h uptrend, strong regime, volume spike
            if close[i] > r3_aligned[i] and uptrend_4h and strong_regime and volume_spike:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S3 in 4h downtrend, strong regime, volume spike
            elif close[i] < s3_aligned[i] and downtrend_4h and strong_regime and volume_spike:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price breaks below S3 or loses 4h uptrend or strong regime
            if close[i] < s3_aligned[i] or not uptrend_4h or not strong_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price breaks above R3 or loses 4h downtrend or strong regime
            if close[i] > r3_aligned[i] or not downtrend_4h or not strong_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals