#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h ADX trend strength + 1-day Donchian breakout with volume confirmation
# Uses 1-day ADX to filter strong trends, 1-day Donchian channels for breakout signals,
# and 12h volume spike for confirmation. Works in bull/bear markets by capturing
# strong directional moves with volume confirmation. Target: 50-150 total trades over 4 years.

name = "12h_ADX_Trend_DailyDonchian_Volume"
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
    
    # Get daily data for ADX and Donchian channels
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 34:  # Need enough for ADX calculation
        return np.zeros(n)
    
    # Calculate daily ADX (14-period)
    high_d = df_daily['high'].values
    low_d = df_daily['low'].values
    close_d = df_daily['close'].values
    
    # True Range
    tr1 = high_d[1:] - low_d[1:]
    tr2 = np.abs(high_d[1:] - close_d[:-1])
    tr3 = np.abs(low_d[1:] - close_d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_d[1:] - high_d[:-1]) > (low_d[:-1] - low_d[1:]), np.maximum(high_d[1:] - high_d[:-1], 0), 0)
    dm_minus = np.where((low_d[:-1] - low_d[1:]) > (high_d[1:] - high_d[:-1]), np.maximum(low_d[:-1] - low_d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Wilder's smoothing function
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        result[period-1] = np.nansum(data[:period]) / period
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
    
    # Calculate daily Donchian channels (20-period)
    high_20 = pd.Series(high_d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 12h volume EMA (20-period) for volume spike detection
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align daily indicators to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_daily, adx)
    high_20_aligned = align_htf_to_ltf(prices, df_daily, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_daily, low_20)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 34)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if (np.isnan(adx_aligned[i]) or np.isnan(high_20_aligned[i]) or 
            np.isnan(low_20_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: current 12h volume > 2.0x 20-period EMA
        vol_filter = volume[i] > 2.0 * vol_ema_20[i]
        
        # ADX filter: > 25 indicates strong trend
        adx_filter = adx_aligned[i] > 25
        
        if position == 0:
            # Look for breakout entry with volume and ADX confirmation
            if close[i] > high_20_aligned[i] and vol_filter and adx_filter:
                signals[i] = 0.25
                position = 1
            elif close[i] < low_20_aligned[i] and vol_filter and adx_filter:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below lower Donchian or ADX weakens
            if close[i] < low_20_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above upper Donchian or ADX weakens
            if close[i] > high_20_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals