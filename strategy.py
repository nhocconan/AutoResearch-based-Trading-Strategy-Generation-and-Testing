#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h ATR-based breakout with 1d trend filter and volume confirmation
# Uses daily ADX to identify trending markets, 4h ATR breakouts for entry, and volume filter.
# Designed to capture strong trends in both bull and bear markets while avoiding chop.
# Target: 25-40 trades/year.

name = "4h_ATRBreakout_1dADX_VolumeFilter"
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
    
    # Get daily data for ADX trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 14:
        return np.zeros(n)
    
    # Calculate daily ADX (14-period)
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # True Range
    tr = np.maximum(high_daily[1:] - low_daily[1:], 
                    np.maximum(np.abs(high_daily[1:] - close_daily[:-1]),
                               np.abs(low_daily[1:] - close_daily[:-1])))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    up_move = high_daily[1:] - high_daily[:-1]
    down_move = low_daily[:-1] - low_daily[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed values
    def smooth_series(arr, period):
        smoothed = np.full_like(arr, np.nan)
        if len(arr) < period:
            return smoothed
        smoothed[period-1] = np.nansum(arr[:period])
        for i in range(period, len(arr)):
            if not np.isnan(smoothed[i-1]):
                smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + arr[i]
        return smoothed
    
    atr = smooth_series(tr, 14)
    plus_di = 100 * smooth_series(plus_dm, 14) / atr
    minus_di = 100 * smooth_series(minus_dm, 14) / atr
    dx = np.where((plus_di + minus_di) > 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = smooth_series(dx, 14)
    
    # Calculate 4h ATR for breakout channels
    tr_4h = np.maximum(high[1:] - low[1:], 
                       np.maximum(np.abs(high[1:] - close[:-1]),
                                  np.abs(low[1:] - close[:-1])))
    tr_4h = np.concatenate([[np.nan], tr_4h])
    
    atr_4h = np.full_like(tr_4h, np.nan)
    if len(tr_4h) >= 10:
        atr_4h[9] = np.nanmean(tr_4h[:10])
        for i in range(10, len(tr_4h)):
            if not np.isnan(atr_4h[i-1]):
                atr_4h[i] = (atr_4h[i-1] * 9 + tr_4h[i]) / 10
    
    # Calculate upper and lower channels (mean ± 1.5 * ATR)
    sma_20 = np.full_like(close, np.nan)
    if len(close) >= 20:
        sma_20[19] = np.mean(close[:20])
        for i in range(20, len(close)):
            sma_20[i] = (sma_20[i-1] * 19 + close[i]) / 20
    
    upper_channel = sma_20 + 1.5 * atr_4h
    lower_channel = sma_20 - 1.5 * atr_4h
    
    # Calculate volume filter (20-period average)
    vol_ma_20 = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma_20[19] = np.mean(volume[:20])
        for i in range(20, len(volume)):
            vol_ma_20[i] = (vol_ma_20[i-1] * 19 + volume[i]) / 20
    
    # Align daily ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_daily, adx)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 10)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(adx_aligned[i]) or np.isnan(sma_20[i]) or 
            np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average
        vol_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        # ADX filter: trending market (ADX > 25)
        trending = adx_aligned[i] > 25
        
        if position == 0:
            # Look for breakout entry with volume and trend confirmation
            long_breakout = close[i] > upper_channel[i] and vol_filter and trending
            short_breakout = close[i] < lower_channel[i] and vol_filter and trending
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
            elif short_breakout:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to middle or trend weakens
            if close[i] < sma_20[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to middle or trend weakens
            if close[i] > sma_20[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals