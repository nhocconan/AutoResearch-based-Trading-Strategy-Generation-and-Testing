#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter and volume confirmation.
Long when price breaks above 6h Donchian upper (20) AND weekly pivot shows bullish bias (close > weekly PP) AND volume > 1.5x average.
Short when price breaks below 6h Donchian lower (20) AND weekly pivot shows bearish bias (close < weekly PP) AND volume > 1.5x average.
Exit when price reverts to 6h Donchian midpoint or weekly pivot reverses.
Uses 6h timeframe for lower trade frequency to minimize fee drag. Weekly pivot provides structural bias from higher timeframe.
Volume confirmation ensures high-conviction breakouts. Works in bull markets via breakouts and in bear markets via short breakdowns with weekly bias filter.
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 6h data for Donchian channels - ONCE before loop
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    close_6h = df_6h['close'].values
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    # Calculate Donchian channels for 6h (20-period)
    highest_20 = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_20 + lowest_20) / 2.0
    
    # Load weekly data for pivot point bias - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly pivot point (standard: PP = (H+L+C)/3)
    weekly_pp = (high_1w + low_1w + close_1w) / 3.0
    
    # Align 6h indicators to 6h timeframe (no alignment needed for same timeframe)
    # But we need to align weekly PP to 6h timeframe
    highest_20_aligned = align_htf_to_ltf(prices, df_6h, highest_20)
    lowest_20_aligned = align_htf_to_ltf(prices, df_6h, lowest_20)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_6h, donchian_mid)
    weekly_pp_aligned = align_htf_to_ltf(prices, df_1w, weekly_pp)
    
    # Volume average (20-period) on 6h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(highest_20_aligned[i]) or np.isnan(lowest_20_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(weekly_pp_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        highest_val = highest_20_aligned[i]
        lowest_val = lowest_20_aligned[i]
        mid_val = donchian_mid_aligned[i]
        weekly_pp_val = weekly_pp_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_current = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper AND weekly close > weekly PP (bullish bias) AND volume spike
            if (price > highest_val and close_6h[-1] > weekly_pp_val if len(close_6h) > 0 else False and vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower AND weekly close < weekly PP (bearish bias) AND volume spike
            elif (price < lowest_val and close_6h[-1] < weekly_pp_val if len(close_6h) > 0 else False and vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price reverts to Donchian midpoint OR weekly close < weekly PP (bias reversal)
                if price <= mid_val or (len(close_6h) > 0 and close_6h[-1] < weekly_pp_val):
                    exit_signal = True
            else:  # position == -1
                # Exit short: price reverts to Donchian midpoint OR weekly close > weekly PP (bias reversal)
                if price >= mid_val or (len(close_6h) > 0 and close_6h[-1] > weekly_pp_val):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Donchian20_WeeklyPivot_Volume"
timeframe = "6h"
leverage = 1.0