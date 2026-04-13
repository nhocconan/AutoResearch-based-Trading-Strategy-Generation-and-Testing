#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout with 12h volume confirmation and 12h chop regime filter
    # Long: price breaks above Donchian(20) high AND 12h volume > 1.3 * 20-period average AND chop > 61.8 (range)
    # Short: price breaks below Donchian(20) low AND 12h volume > 1.3 * 20-period average AND chop > 61.8 (range)
    # Exit: price reverts to Donchian(20) midpoint OR chop < 38.2 (trending)
    # Using 12h for volume/chop to reduce noise and avoid overtrading, 4h for price action
    # Discrete position sizing (0.25) to minimize fee churn
    # Target: 15-35 trades/year (~60-140 over 4 years) to stay within fee drag limits
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian channels (call ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Get 12h data for volume and chop (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Donchian high: rolling max of high
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    # Donchian low: rolling min of low
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    # Donchian midpoint: average of high and low
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Align 4h Donchian to 4h timeframe (no additional delay needed for price channels)
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_4h, donchian_mid)
    
    # Calculate 12h volume spike filter: volume > 1.3 * 20-period average
    volume_12h = df_12h['volume'].values
    vol_ma_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_12h > (1.3 * vol_ma_20)
    
    # Calculate 12h Choppiness Index (CHOP) - range/trend regime filter
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range for 12h
    tr1 = np.abs(high_12h[1:] - low_12h[1:])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
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
    
    atr_12h = wilders_smoothing(tr, 14)
    
    # Choppiness Index (14-period)
    chop_period = 14
    sum_atr = np.full_like(atr_12h, np.nan)
    highest_high = np.full_like(high_12h, np.nan)
    lowest_low = np.full_like(low_12h, np.nan)
    
    for i in range(len(atr_12h)):
        if i < chop_period - 1:
            continue
        if np.isnan(atr_12h[i-chop_period+1:i+1]).any():
            continue
        sum_atr[i] = np.nansum(atr_12h[i-chop_period+1:i+1])
        highest_high[i] = np.nanmax(high_12h[i-chop_period+1:i+1])
        lowest_low[i] = np.nanmin(low_12h[i-chop_period+1:i+1])
    
    # Avoid division by zero
    range_12h = highest_high - lowest_low
    chop = np.full_like(atr_12h, 50.0)  # default to neutral
    mask = (range_12h > 0) & ~np.isnan(sum_atr)
    chop[mask] = 100 * np.log10(sum_atr[mask] / (np.log10(chop_period) * range_12h[mask]))
    
    # Align 12h indicators to 4h (wait for completed 12h bar)
    volume_spike_aligned = align_htf_to_ltf(prices, df_12h, volume_spike.astype(float))
    chop_aligned = align_htf_to_ltf(prices, df_12h, chop)
    
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
        
        # Volume confirmation: 12h volume spike
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

name = "4h_12h_donchian_volume_chop_regime_v1"
timeframe = "4h"
leverage = 1.0