#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h strategy using 1d Donchian breakout with 1w ADX trend filter and volume confirmation
    # Long: price breaks above 1d Donchian(20) high AND 1w ADX > 25 (trending) AND volume > 1.5x 20-period average
    # Short: price breaks below 1d Donchian(20) low AND 1w ADX > 25 AND volume > 1.5x average
    # Exit: price reverts to 1d Donchian midpoint OR ADX < 20 (trend weakening) OR volume dry-up
    # Using 6h timeframe for balance of trade frequency and noise reduction.
    # Donchian provides objective breakout levels, ADX filters for trending markets, volume confirms conviction.
    # Discrete position sizing (0.25) to minimize fee churn.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get weekly data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 1d Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Donchian high: highest high over 20 periods
    donch_high = np.full(len(high_1d), np.nan)
    for i in range(20, len(high_1d)):
        donch_high[i] = np.max(high_1d[i-20:i])
    
    # Donchian low: lowest low over 20 periods
    donch_low = np.full(len(low_1d), np.nan)
    for i in range(20, len(low_1d)):
        donch_low[i] = np.min(low_1d[i-20:i])
    
    # Donchian midpoint: average of high and low
    donch_mid = (donch_high + donch_low) / 2
    
    # Align Donchian levels to 6h
    donch_high_6h = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_6h = align_htf_to_ltf(prices, df_1d, donch_low)
    donch_mid_6h = align_htf_to_ltf(prices, df_1d, donch_mid)
    
    # Calculate 1w ADX(14) for trend filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = np.abs(high_1w[1:] - low_1w[1:])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # prepend NaN for first element
    
    # Directional Movement
    dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    
    dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Wilder's smoothing (14-period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[1:period])
        # Subsequent values
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr = wilders_smoothing(tr, 14)
    dmp = wilders_smoothing(dm_plus, 14)
    dmm = wilders_smoothing(dm_minus, 14)
    
    # Directional Indicators
    di_plus = np.where(atr != 0, 100 * dmp / atr, 0)
    di_minus = np.where(atr != 0, 100 * dmm / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smoothing(dx, 14)
    
    # Align ADX to 6h
    adx_6h = align_htf_to_ltf(prices, df_1w, adx)
    
    # Volume confirmation: >1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donch_high_6h[i]) or np.isnan(donch_low_6h[i]) or 
            np.isnan(donch_mid_6h[i]) or np.isnan(adx_6h[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: ADX > 25 = strong trend
        trending = adx_6h[i] > 25
        weak_trend = adx_6h[i] < 20  # exit condition
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        # Breakout conditions
        long_breakout = close[i] > donch_high_6h[i]
        short_breakout = close[i] < donch_low_6h[i]
        
        # Exit conditions: price reverts to midpoint OR trend weakens OR volume dry-up
        long_exit = (close[i] < donch_mid_6h[i]) or weak_trend or not vol_confirm
        short_exit = (close[i] > donch_mid_6h[i]) or weak_trend or not vol_confirm
        
        if long_breakout and trending and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_breakout and trending and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_1w_donchian_adx_volume_v1"
timeframe = "6h"
leverage = 1.0