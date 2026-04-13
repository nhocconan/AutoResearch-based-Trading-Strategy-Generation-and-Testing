#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout with 12h volume spike and 12h Chop regime filter
    # Long: price > Donchian(20) high AND 12h volume > 2.0 * 20-period avg volume AND 12h Chop > 61.8 (ranging)
    # Short: price < Donchian(20) low AND 12h volume > 2.0 * 20-period avg volume AND 12h Chop > 61.8 (ranging)
    # Exit: price crosses Donchian midpoint OR Chop < 38.2 (trending market begins)
    # Uses discrete position sizing (0.25) to minimize fee churn
    # Target: 75-150 total trades over 4 years (~19-38/year) to stay within limits
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian channels (call ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Get 12h data for volume and chop confirmation (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
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
    
    highest_high = rolling_max(high_4h, 20)
    lowest_low = rolling_min(low_4h, 20)
    donchian_high = highest_high
    donchian_low = lowest_low
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Align 4h Donchian to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_4h, donchian_mid)
    
    # Calculate 12h volume average (20-period)
    volume_12h = df_12h['volume'].values
    def rolling_mean(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(len(arr)):
            if i >= window - 1:
                result[i] = np.mean(arr[i-window+1:i+1])
        return result
    
    vol_ma_20 = rolling_mean(volume_12h, 20)
    vol_ma_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20)
    
    # Calculate 12h Chop Index (Choppiness Index)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = np.abs(high_12h[1:] - low_12h[1:])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    
    # ATR (14-period) using Wilder's smoothing
    def wilders_smoothing(data, period):
        alpha = 1.0 / period
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nanmean(data[:period])
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_12h = wilders_smoothing(tr, 14)
    
    # Sum of True Range over 14 periods
    def rolling_sum(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(len(arr)):
            if i >= window - 1:
                result[i] = np.sum(arr[i-window+1:i+1])
        return result
    
    sum_tr_14 = rolling_sum(tr, 14)
    
    # Chop Index = 100 * log10(sum(TR14) / (ATR14 * 14)) / log10(14)
    chop = np.where((atr_12h * 14) > 0,
                    100 * np.log10(sum_tr_14 / (atr_12h * 14)) / np.log10(14),
                    np.nan)
    
    # Align 12h Chop to 12h (wait for completed 12h bar)
    chop_aligned = align_htf_to_ltf(prices, df_12h, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(vol_ma_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade when 12h Chop > 61.8 (ranging market)
        ranging_market = chop_aligned[i] > 61.8
        # Exit regime: Chop < 38.2 (trending market begins)
        trending_market = chop_aligned[i] < 38.2
        
        # Volume confirmation: current 12h volume > 2.0 * 20-period average
        vol_12h_current = df_12h['volume'].values
        vol_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_12h_current)
        volume_confirm = vol_12h_aligned[i] > 2.0 * vol_ma_aligned[i]
        
        # Donchian breakout signals
        long_breakout = close[i] > donchian_high_aligned[i]
        short_breakout = close[i] < donchian_low_aligned[i]
        
        # Entry logic: Donchian breakout + volume confirmation + ranging market
        long_entry = long_breakout and volume_confirm and ranging_market
        short_entry = short_breakout and volume_confirm and ranging_market
        
        # Exit logic: price crosses Donchian midpoint OR market becomes trending
        long_exit = close[i] < donchian_mid_aligned[i] or trending_market
        short_exit = close[i] > donchian_mid_aligned[i] or trending_market
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
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

name = "4h_12h_donchian_volume_chop_v2"
timeframe = "4h"
leverage = 1.0