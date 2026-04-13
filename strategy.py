#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Williams %R mean reversion + 1d volume spike + 1w chop regime filter
    # Long: Williams %R(14) < -80 AND 1d volume > 2.0 * 20-period avg volume AND 1w Chop > 61.8 (ranging market)
    # Short: Williams %R(14) > -20 AND 1d volume > 2.0 * 20-period avg volume AND 1w Chop > 61.8 (ranging market)
    # Exit: Williams %R crosses -50 OR Chop < 38.2 (trending market begins)
    # Uses 12h for Williams %R, 1d for volume confirmation, 1w for chop regime
    # Discrete position sizing (0.25) to minimize fee churn
    # Target: 50-150 total trades over 4 years (~12-37/year) to stay within limits
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Williams %R (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Get 1d data for volume confirmation (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get 1w data for Chop regime filter (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 12h Williams %R (14-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
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
    
    highest_high = rolling_max(high_12h, 14)
    lowest_low = rolling_min(low_12h, 14)
    williams_r = np.where((highest_high - lowest_low) != 0,
                          (highest_high - close_12h) / (highest_high - lowest_low) * -100,
                          np.nan)
    
    # Align 12h Williams %R to 12h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_12h, williams_r)
    
    # Calculate 1d volume average (20-period)
    volume_1d = df_1d['volume'].values
    def rolling_mean(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(len(arr)):
            if i >= window - 1:
                result[i] = np.mean(arr[i-window+1:i+1])
        return result
    
    vol_ma_20 = rolling_mean(volume_1d, 20)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Calculate 1w Chop Index (Choppiness Index)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = np.abs(high_1w[1:] - low_1w[1:])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
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
    
    atr_1w = wilders_smoothing(tr, 14)
    
    # Sum of True Range over 14 periods
    def rolling_sum(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(len(arr)):
            if i >= window - 1:
                result[i] = np.sum(arr[i-window+1:i+1])
        return result
    
    sum_tr_14 = rolling_sum(tr, 14)
    
    # Chop Index = 100 * log10(sum(TR14) / (ATR14 * 14)) / log10(14)
    chop = np.where((atr_1w * 14) > 0,
                    100 * np.log10(sum_tr_14 / (atr_1w * 14)) / np.log10(14),
                    np.nan)
    
    # Align 1w Chop to 1w (wait for completed 1w bar)
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(vol_ma_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade when 1w Chop > 61.8 (ranging market)
        ranging_market = chop_aligned[i] > 61.8
        # Exit regime: Chop < 38.2 (trending market begins)
        trending_market = chop_aligned[i] < 38.2
        
        # Volume confirmation: current 1d volume > 2.0 * 20-period average
        # Get current 1d volume (need to align 1d volume to 12h timeframe)
        vol_1d_current = df_1d['volume'].values
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d_current)
        volume_confirm = vol_1d_aligned[i] > 2.0 * vol_ma_aligned[i]
        
        # Williams %R signals
        oversold = williams_r_aligned[i] < -80
        overbought = williams_r_aligned[i] > -20
        
        # Entry logic: Williams %R extreme + volume confirmation + ranging market
        long_entry = oversold and volume_confirm and ranging_market
        short_entry = overbought and volume_confirm and ranging_market
        
        # Exit logic: Williams %R crosses -50 OR market becomes trending
        long_exit = williams_r_aligned[i] > -50 or trending_market
        short_exit = williams_r_aligned[i] < -50 or trending_market
        
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

name = "12h_1d_1w_williamsr_volume_chop_v1"
timeframe = "12h"
leverage = 1.0