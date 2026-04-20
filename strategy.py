#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Adaptive_Range_Breakout_Volume_Confirm"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily ADX for trend strength (14-period)
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
    up_move = np.diff(high_1d)
    down_move = -np.diff(low_1d)
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed values
    def smooth_wilder(arr, period):
        smoothed = np.full_like(arr, np.nan)
        if len(arr) < period:
            return smoothed
        smoothed[period-1] = np.nansum(arr[:period])
        for i in range(period, len(arr)):
            smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + arr[i]
        return smoothed
    
    atr_1d = smooth_wilder(tr, 14)
    plus_di_1d = 100 * smooth_wilder(plus_dm, 14) / atr_1d
    minus_di_1d = 100 * smooth_wilder(minus_dm, 14) / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = smooth_wilder(dx_1d, 14)
    
    # Daily range for range calculation
    daily_range = high_1d - low_1d
    range_ma = np.full_like(daily_range, np.nan)
    for i in range(19, len(daily_range)):
        range_ma[i] = np.mean(daily_range[i-19:i+1])
    
    # Align ADX and range MA to 12h
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    range_ma_aligned = align_htf_to_ltf(prices, df_1d, range_ma)
    
    # Daily average volume for spike detection
    vol_1d = df_1d['volume'].values
    vol_avg_1d = np.full_like(vol_1d, np.nan)
    for i in range(19, len(vol_1d)):
        vol_avg_1d[i] = np.mean(vol_1d[i-19:i+1])
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # Calculate ATR for stop loss (14-period on 12h data)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr = np.full_like(tr, np.nan)
    for i in range(13, len(tr)):
        atr[i] = np.nanmean(tr[i-13:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 20  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Only trade when ADX < 25 (range-bound market)
        if np.isnan(adx_1d_aligned[i]) or adx_1d_aligned[i] >= 25:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Get previous completed daily bar for range calculation
        if len(df_1d) < 2:
            continue
            
        # Calculate daily range-based levels for previous day
        j = len(df_1d) - 1  # Previous completed day
        if j < 1:
            continue
            
        prev_high = high_1d[j-1]
        prev_low = low_1d[j-1]
        prev_close = close_1d[j-1]
        prev_range = prev_high - prev_low
        
        if prev_range <= 0:
            continue
            
        # Adaptive range levels: 38.2% and 61.8% of range from low
        range_level_low = prev_low + 0.382 * prev_range
        range_level_high = prev_low + 0.618 * prev_range
        
        current_close = prices['close'].iloc[i]
        current_volume = prices['volume'].iloc[i]
        current_atr = atr[i]
        
        # Volume spike: current volume > 1.5x daily average volume
        vol_spike = (not np.isnan(vol_avg_1d_aligned[i]) and 
                     current_volume > 1.5 * vol_avg_1d_aligned[i])
        
        if position == 0:
            # Long: price breaks above 61.8% level with volume spike in ranging market
            if (current_close > range_level_high and vol_spike):
                signals[i] = 0.25
                position = 1
                entry_price = current_close
            # Short: price breaks below 38.2% level with volume spike in ranging market
            elif (current_close < range_level_low and vol_spike):
                signals[i] = -0.25
                position = -1
                entry_price = current_close
        
        elif position == 1:
            # Long exit: price breaks below 38.2% level or ATR stop loss
            if current_close < range_level_low:
                signals[i] = 0.0
                position = 0
            elif current_atr > 0 and current_close < entry_price - 2.0 * current_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above 61.8% level or ATR stop loss
            if current_close > range_level_high:
                signals[i] = 0.0
                position = 0
            elif current_atr > 0 and current_close > entry_price + 2.0 * current_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals