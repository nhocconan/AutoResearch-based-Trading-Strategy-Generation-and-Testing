#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1d Donchian(20) breakout + 1w volume spike + 1w chop regime filter
    # Long: price breaks above Donchian(20) high AND 1w volume > 1.5 * 20-period average AND chop > 61.8 (range)
    # Short: price breaks below Donchian(20) low AND 1w volume > 1.5 * 20-period average AND chop > 61.8 (range)
    # Exit: price reverts to Donchian(20) midpoint OR chop < 38.2 (trending)
    # Using 1d for price action (primary timeframe) and 1w for filters to reduce noise
    # Discrete position sizing (0.25) to minimize fee churn
    # Target: 30-100 trades over 4 years (~7-25/year) to stay within fee drag limits
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for Donchian channels (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get 1w data for volume and chop filters (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 1d Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian high: rolling max of high
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    # Donchian low: rolling min of low
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    # Donchian midpoint: average of high and low
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Align 1d Donchian to 1d timeframe (no additional delay needed for price channels)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1d, donchian_mid)
    
    # Calculate 1w volume spike filter: volume > 1.5 * 20-period average
    volume_1w = df_1w['volume'].values
    vol_ma_20 = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1w > (1.5 * vol_ma_20)
    
    # Calculate 1w Choppiness Index (CHOP) - range/trend regime filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range for 1w
    tr1 = np.abs(high_1w[1:] - low_1w[1:])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    
    # ATR(14) - using Wilder's smoothing
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
    
    # Choppiness Index (14-period)
    chop_period = 14
    sum_atr = np.full_like(atr_1w, np.nan)
    highest_high = np.full_like(high_1w, np.nan)
    lowest_low = np.full_like(low_1w, np.nan)
    
    for i in range(len(atr_1w)):
        if i < chop_period - 1:
            continue
        if np.isnan(atr_1w[i-chop_period+1:i+1]).any():
            continue
        sum_atr[i] = np.nansum(atr_1w[i-chop_period+1:i+1])
        highest_high[i] = np.nanmax(high_1w[i-chop_period+1:i+1])
        lowest_low[i] = np.nanmin(low_1w[i-chop_period+1:i+1])
    
    # Avoid division by zero
    range_1w = highest_high - lowest_low
    chop = np.full_like(atr_1w, 50.0)  # default to neutral
    mask = (range_1w > 0) & ~np.isnan(sum_atr)
    chop[mask] = 100 * np.log10(sum_atr[mask] / (np.log10(chop_period) * range_1w[mask]))
    
    # Align 1w indicators to 1d (wait for completed 1w bar)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1w, volume_spike.astype(float))
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(volume_spike_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade when chop > 61.8 (range-bound market)
        in_range = chop_aligned[i] > 61.8
        # Exit regime: chop < 38.2 (trending market) - exit positions
        in_trend = chop_aligned[i] < 38.2
        
        # Volume confirmation: 1w volume spike
        vol_confirmed = volume_spike_aligned[i] > 0.5  # boolean as float
        
        # Breakout conditions
        long_breakout = close[i] > donchian_high_aligned[i]
        short_breakout = close[i] < donchian_low_aligned[i]
        
        # Entry logic: Donchian breakout + volume spike + range regime
        long_entry = long_breakout and vol_confirmed and in_range
        short_entry = short_breakout and vol_confirmed and in_range
        
        # Exit logic: price reverts to midpoint OR regime shifts to trending
        long_exit = (close[i] < donchian_mid_aligned[i]) or in_trend
        short_exit = (close[i] > donchian_mid_aligned[i]) or in_trend
        
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

name = "1d_1w_donchian_volume_chop_regime_v1"
timeframe = "1d"
leverage = 1.0