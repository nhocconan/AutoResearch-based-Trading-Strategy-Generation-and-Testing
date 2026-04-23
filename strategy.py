#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter and volume confirmation.
Long when price breaks above 6h Donchian upper (20-period high) AND weekly pivot shows bullish bias (close > weekly pivot) AND volume > 1.5x average.
Short when price breaks below 6h Donchian lower (20-period low) AND weekly pivot shows bearish bias (close < weekly pivot) AND volume > 1.5x average.
Weekly pivot provides higher-timeframe structure to avoid counter-trend trades in both bull and bear markets.
Designed for low trade frequency (target: 50-150 total trades over 4 years) with high conviction signals.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 6h data for Donchian calculation - ONCE before loop
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    # Calculate Donchian channels from previous 6h bar (avoid look-ahead)
    prev_high = np.roll(high_6h, 1)
    prev_low = np.roll(low_6h, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    # 20-period Donchian channels
    upper_20 = pd.Series(prev_high).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(prev_low).rolling(window=20, min_periods=20).min().values
    
    # Load 1d data for weekly pivot calculation - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot from previous daily bar (avoid look-ahead)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    prev_close_1d[0] = np.nan
    
    # Weekly pivot point (using daily data as proxy for weekly structure)
    pivot_1d = (prev_high_1d + prev_low_1d + prev_close_1d) / 3.0
    
    # Load 1w data for weekly trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Simple weekly trend: price above/below weekly EMA21
    ema21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align HTF indicators to 6h timeframe
    upper_20_aligned = align_htf_to_ltf(prices, df_6h, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_6h, lower_20)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    ema21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w)
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or 
            np.isnan(pivot_1d_aligned[i]) or np.isnan(ema21_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        upper_val = upper_20_aligned[i]
        lower_val = lower_20_aligned[i]
        pivot_val = pivot_1d_aligned[i]
        ema21_1w_val = ema21_1w_aligned[i]
        vol_ma_val = vol_ma[i]
        price = close[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper AND weekly pivot bullish (close > pivot) AND weekly uptrend (close > weekly EMA21) AND volume spike
            if (price > upper_val and price > pivot_val and price > ema21_1w_val and vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below Donchian lower AND weekly pivot bearish (close < pivot) AND weekly downtrend (close < weekly EMA21) AND volume spike
            elif (price < lower_val and price < pivot_val and price < ema21_1w_val and vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below Donchian lower OR weekly trend turns bearish
                if (price < lower_val or price < ema21_1w_val):
                    exit_signal = True
            else:  # position == -1
                # Exit short: price breaks above Donchian upper OR weekly trend turns bullish
                if (price > upper_val or price > ema21_1w_val):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Donchian20_WeeklyPivot_VolumeSpike"
timeframe = "6h"
leverage = 1.0