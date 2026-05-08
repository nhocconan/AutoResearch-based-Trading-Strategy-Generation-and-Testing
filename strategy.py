#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d ADX trend filter and volume confirmation
# Uses 1d ADX > 25 to identify trending markets, 4h Donchian(20) breakout for entry,
# and volume > 1.5x 20-period average for confirmation. Works in both bull/bear by
# following the trend direction from ADX. Target: 25-40 trades/year.

name = "4h_Donchian_ADX_Volume"
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
    if len(df_daily) < 30:
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
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nansum(data[:period])
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr_14 = wilders_smooth(tr, 14)
    plus_di_14 = 100 * wilders_smooth(plus_dm, 14) / atr_14
    minus_di_14 = 100 * wilders_smooth(minus_dm, 14) / atr_14
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)
    adx = wilders_smooth(dx, 14)
    
    # Pre-calculate Donchian channels (20-period) for 4h
    def rolling_max(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(len(arr)):
            if i >= window - 1:
                result[i] = np.max(arr[i-window+1:i+1])
        return result
    
    def rolling_min(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(len(arr)):
            if i >= window - 1:
                result[i] = np.min(arr[i-window+1:i+1])
        return result
    
    donchian_high = rolling_max(high, 20)
    donchian_low = rolling_min(low, 20)
    
    # Volume average (20-period)
    vol_avg_20 = np.full_like(volume, np.nan)
    for i in range(len(volume)):
        if i >= 19:
            vol_avg_20[i] = np.mean(volume[i-19:i+1])
    
    # Align daily ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_daily, adx)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(adx_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_avg_20[i]
        
        if position == 0:
            # Look for entry: Donchian breakout in trending market (ADX > 25)
            trending = adx_aligned[i] > 25
            
            # Long when price breaks above Donchian high
            long_condition = (
                close[i] > donchian_high[i] and   # breakout above upper band
                trending and                      # trending market
                vol_confirm                       # volume confirmation
            )
            
            # Short when price breaks below Donchian low
            short_condition = (
                close[i] < donchian_low[i] and    # breakout below lower band
                trending and                      # trending market
                vol_confirm                       # volume confirmation
            )
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns below Donchian low or trend weakens
            if close[i] < donchian_low[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above Donchian high or trend weakens
            if close[i] > donchian_high[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals