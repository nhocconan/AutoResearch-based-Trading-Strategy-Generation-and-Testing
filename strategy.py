#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 breakout with 1d volume spike and ADX regime filter
- Uses Camarilla pivot levels (R3/S3) from 1d for high-probability breakout signals
- 1d volume > 1.5x 20-period average confirms institutional participation
- ADX(14) > 25 on 4h ensures trending market to avoid chop whipsaws
- Designed for 4h timeframe targeting 20-50 trades/year (80-200 over 4 years)
- Works in bull markets (breakouts continuation) and bear markets (breakdown continuation)
- Volume spike and ADX filter reduce false breakouts during low-participation periods
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d Camarilla pivot levels (R3, S3, pivot)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = np.concatenate([[np.nan], high_1d[:-1]])
    prev_low = np.concatenate([[np.nan], low_1d[:-1]])
    prev_close = np.concatenate([[np.nan], close_1d[:-1]])
    
    # Pivot point
    pivot = (prev_high + prev_low + prev_close) / 3.0
    # Range
    range_1d = prev_high - prev_low
    
    # Camarilla levels: R3 = close + (high-low)*1.1/4, S3 = close - (high-low)*1.1/4
    camarilla_r3 = prev_close + (range_1d * 1.1 / 4.0)
    camarilla_s3 = prev_close - (range_1d * 1.1 / 4.0)
    
    # Align Camarilla levels to 1d timeframe (already completed bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate 4h ADX regime filter (ADX > 25 = trending market)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr1 = np.abs(high_4h[1:] - low_4h[1:])
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    up_move = np.concatenate([[np.nan], high_4h[1:] - high_4h[:-1]])
    down_move = np.concatenate([[np.nan], low_4h[:-1] - low_4h[1:]])
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed TR, +DM, -DM (Wilder's smoothing)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[:period])
        # Subsequent values
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_4h = wilders_smoothing(tr, 14)
    plus_dm_smooth = wilders_smoothing(plus_dm, 14)
    minus_dm_smooth = wilders_smoothing(minus_dm, 14)
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr_4h
    minus_di = 100 * minus_dm_smooth / atr_4h
    
    # DX and ADX
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
    dx = np.where((plus_di + minus_di) == 0, 0, dx)
    
    def adaptive_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        result[period-1] = np.nanmean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    adx = adaptive_smoothing(dx, 14)
    adx_aligned = align_htf_to_ltf(prices, df_4h, adx)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(30, 20)  # ADX, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine breakout conditions
        price_above_r3 = close[i] > camarilla_r3_aligned[i]
        price_below_s3 = close[i] < camarilla_s3_aligned[i]
        
        # Trending regime filter
        trending = adx_aligned[i] > 25
        
        if position == 0:
            # Long conditions: price breaks above R3, trending market, volume spike
            long_signal = (price_above_r3 and 
                          trending and
                          volume[i] > 1.5 * vol_ma[i])
            
            # Short conditions: price breaks below S3, trending market, volume spike
            short_signal = (price_below_s3 and 
                           trending and
                           volume[i] > 1.5 * vol_ma[i])
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: opposite Camarilla breakout or trend weakening
            exit_signal = False
            
            if position == 1:
                # Exit long: price falls below S3 or trend weakens
                if (price_below_s3 or 
                    adx_aligned[i] < 20):  # Trend weakening
                    exit_signal = True
            elif position == -1:
                # Exit short: price rises above R3 or trend weakens
                if (price_above_r3 or 
                    adx_aligned[i] < 20):  # Trend weakening
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dVolumeSpike_ADXTrend"
timeframe = "4h"
leverage = 1.0