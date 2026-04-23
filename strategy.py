#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian breakout with 1d EMA200 trend filter and volume confirmation.
Long when price breaks above 4h Donchian upper channel (20-period) AND price > 1d EMA200 (uptrend) AND volume > 1.5x average.
Short when price breaks below 4h Donchian lower channel (20-period) AND price < 1d EMA200 (downtrend) AND volume > 1.5x average.
Exit when price reverts to 4h Donchian middle (median) or trend reverses (price crosses 1d EMA200).
Uses 4h timeframe with proven structure (Donchian + trend + volume) to limit trades and avoid overtrading.
Target: 50-150 trades over 4 years (12-38/year) to stay within proven working range.
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
    
    # Calculate 4h Donchian channels (20-period) - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Donchian upper/lower (20-period high/low of previous 20 bars)
    donchian_upper = pd.Series(high_4h).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_lower = pd.Series(low_4h).rolling(window=20, min_periods=20).min().shift(1).values
    donchian_middle = (donchian_upper + donchian_lower) / 2.0
    
    # Load 1d data for EMA200 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align HTF indicators to 4h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_4h, donchian_middle)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Volume average (20-period) on primary timeframe
    vol_ma_primary = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or np.isnan(donchian_middle_aligned[i]) or 
            np.isnan(ema200_1d_aligned[i]) or np.isnan(vol_ma_primary[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        upper_val = donchian_upper_aligned[i]
        lower_val = donchian_lower_aligned[i]
        middle_val = donchian_middle_aligned[i]
        ema200_val = ema200_1d_aligned[i]
        vol_ma_val = vol_ma_primary[i]
        
        # Get current price and volume
        price = close[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: price breaks above 4h Donchian upper AND price > 1d EMA200 (uptrend) AND volume confirmation
            if (price > upper_val and price > ema200_val and vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 4h Donchian lower AND price < 1d EMA200 (downtrend) AND volume confirmation
            elif (price < lower_val and price < ema200_val and vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price reverts to Donchian middle OR price breaks below 1d EMA200 (trend reversal)
                if price <= middle_val or price < ema200_val:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price reverts to Donchian middle OR price breaks above 1d EMA200 (trend reversal)
                if price >= middle_val or price > ema200_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian20_1dEMA200_Volume"
timeframe = "4h"
leverage = 1.0