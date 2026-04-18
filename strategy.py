#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with daily ADX filter and volume confirmation.
# Donchian channels capture breakouts from price consolidation.
# Daily ADX > 25 filters for trending markets only, avoiding chop.
# Volume confirmation ensures breakout conviction.
# Designed for low trade frequency (20-50/year) to minimize fee drag in 4h timeframe.
# Works in bull markets (breakouts above upper band) and bear markets (breakouts below lower band).
name = "4h_Donchian20_DailyADX_Volume_Filter"
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
    
    # Get daily data for ADX filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Donchian channels (20-period) using previous period's data to avoid look-ahead
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    upper_band = high_20
    lower_band = low_20
    
    # Calculate daily ADX (14-period)
    high_d = df_1d['high'].values
    low_d = df_1d['low'].values
    close_d = df_1d['close'].values
    
    # True Range
    tr1 = high_d[1:] - low_d[1:]
    tr2 = np.abs(high_d[1:] - close_d[:-1])
    tr3 = np.abs(low_d[1:] - close_d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_d[1:] - high_d[:-1]) > (low_d[:-1] - low_d[1:]), 
                       np.maximum(high_d[1:] - high_d[:-1], 0), 0)
    dm_minus = np.where((low_d[:-1] - low_d[1:]) > (high_d[1:] - high_d[:-1]), 
                        np.maximum(low_d[:-1] - low_d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR, DM+, DM- using Wilder's smoothing (EMA with alpha=1/14)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nanmean(data[:period])
            for i in range(period, len(data)):
                if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                    result[i] = result[i-1] * (1 - 1/period) + data[i] * (1/period)
                else:
                    result[i] = np.nan
        return result
    
    tr_smooth = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(tr_smooth != 0, 100 * dm_plus_smooth / tr_smooth, 0)
    di_minus = np.where(tr_smooth != 0, 100 * dm_minus_smooth / tr_smooth, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smoothing(dx, 14)
    
    # ADX threshold for trending market
    adx_threshold = 25
    adx_trending = adx >= adx_threshold
    
    # Align daily ADX to 4h timeframe
    adx_trending_aligned = align_htf_to_ltf(prices, df_1d, adx_trending)
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hour_index = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        hour = hour_index[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma_20[i]
        
        # Trend filter: daily ADX indicates trending market
        trend_filter = adx_trending_aligned[i] if not np.isnan(adx_trending_aligned[i]) else False
        
        if position == 0:
            # Long: price breaks above upper band AND volume confirmation AND trend filter
            long_breakout = close[i] > upper_band[i]
            if vol_confirm and trend_filter and long_breakout:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower band AND volume confirmation AND trend filter
            elif vol_confirm and trend_filter and close[i] < lower_band[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls below lower band OR ADX drops below threshold (trend weakening)
            exit_condition = close[i] < lower_band[i] or (np.isnan(adx_trending_aligned[i]) or not adx_trending_aligned[i])
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above upper band OR ADX drops below threshold (trend weakening)
            exit_condition = close[i] > upper_band[i] or (np.isnan(adx_trending_aligned[i]) or not adx_trending_aligned[i])
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals