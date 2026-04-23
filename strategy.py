#!/usr/bin/env python3
"""
Hypothesis: 4h strategy using 12h Supertrend as trend filter, Donchian(20) breakout on 4h for entry,
and ATR(14) volatility filter. Long when price breaks above 4h Donchian upper band AND 12h Supertrend is bullish
AND ATR ratio > 0.8 (avoid low volatility chop). Short when price breaks below 4h Donchian lower band
AND 12h Supertrend is bearish AND ATR ratio > 0.8. Exit when price reverts to 4h Donchian midpoint
or Supertrend flips. Uses discrete position sizing (0.25) to limit fee churn. Target: 50-120 trades over 4 years.
Works in bull via breakouts, in bear via short breakdowns with trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 4h Donchian channels (20-period) - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Donchian channels on 4h (based on previous 20 completed 4h bars)
    def rolling_max(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            res[i] = np.max(arr[i-window+1:i+1])
        return res
    
    def rolling_min(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            res[i] = np.min(arr[i-window+1:i+1])
        return res
    
    donchian_upper_4h = rolling_max(high_4h, 20)
    donchian_lower_4h = rolling_min(low_4h, 20)
    donchian_mid_4h = (donchian_upper_4h + donchian_lower_4h) / 2.0
    
    # Shift by 1 to avoid look-ahead (use previous bar's channel)
    donchian_upper_4h = np.roll(donchian_upper_4h, 1)
    donchian_lower_4h = np.roll(donchian_lower_4h, 1)
    donchian_mid_4h = np.roll(donchian_mid_4h, 1)
    donchian_upper_4h[0] = np.nan
    donchian_lower_4h[0] = np.nan
    donchian_mid_4h[0] = np.nan
    
    # Load 12h data for Supertrend trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate ATR(10) for Supertrend
    def true_range(high, low, close):
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr1[0] = np.nan
        tr2[0] = np.nan
        tr3[0] = np.nan
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    tr_12h = true_range(high_12h, low_12h, close_12h)
    atr_12h = pd.Series(tr_12h).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Supertrend parameters
    atr_multiplier = 3.0
    upper_band_12h = (high_12h + low_12h) / 2 + atr_multiplier * atr_12h
    lower_band_12h = (high_12h + low_12h) / 2 - atr_multiplier * atr_12h
    
    # Initialize Supertrend
    supertrend_12h = np.full_like(close_12h, np.nan)
    direction_12h = np.ones_like(close_12h)  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(close_12h)):
        if np.isnan(atr_12h[i]) or np.isnan(upper_band_12h[i]) or np.isnan(lower_band_12h[i]):
            continue
            
        if i == 1:
            supertrend_12h[i] = lower_band_12h[i]
            direction_12h[i] = 1
            continue
            
        prev_supertrend = supertrend_12h[i-1]
        prev_direction = direction_12h[i-1]
        
        if close_12h[i] > upper_band_12h[i-1]:
            direction_12h[i] = 1
        elif close_12h[i] < lower_band_12h[i-1]:
            direction_12h[i] = -1
        else:
            direction_12h[i] = prev_direction
            
        if direction_12h[i] == 1:
            supertrend_12h[i] = max(lower_band_12h[i], prev_supertrend)
        else:
            supertrend_12h[i] = min(upper_band_12h[i], prev_supertrend)
    
    # ATR ratio filter (4h ATR(14) / 4h ATR(50)) to avoid low volatility regimes
    tr_4h = true_range(high_4h, low_4h, close_4h)
    atr_14_4h = pd.Series(tr_4h).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_50_4h = pd.Series(tr_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    atr_ratio_4h = atr_14_4h / (atr_50_4h + 1e-10)  # Avoid division by zero
    
    # Align HTF indicators to 4h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper_4h)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower_4h)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_4h, donchian_mid_4h)
    supertrend_12h_aligned = align_htf_to_ltf(prices, df_12h, supertrend_12h)
    direction_12h_aligned = align_htf_to_ltf(prices, df_12h, direction_12h)
    atr_ratio_aligned = align_htf_to_ltf(prices, df_4h, atr_ratio_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(supertrend_12h_aligned[i]) or 
            np.isnan(direction_12h_aligned[i]) or np.isnan(atr_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        upper_val = donchian_upper_aligned[i]
        lower_val = donchian_lower_aligned[i]
        mid_val = donchian_mid_aligned[i]
        supertrend_val = supertrend_12h_aligned[i]
        direction_val = direction_12h_aligned[i]
        atr_ratio_val = atr_ratio_aligned[i]
        
        # Get current price
        price = close[i]
        
        if position == 0:
            # Long: price breaks above 4h Donchian upper AND 12h Supertrend bullish AND sufficient volatility
            if (price > upper_val and direction_val == 1 and atr_ratio_val > 0.8):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 4h Donchian lower AND 12h Supertrend bearish AND sufficient volatility
            elif (price < lower_val and direction_val == -1 and atr_ratio_val > 0.8):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price reverts to Donchian mid OR Supertrend turns bearish
                if price <= mid_val or direction_val == -1:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price reverts to Donchian mid OR Supertrend turns bullish
                if price >= mid_val or direction_val == 1:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Supertrend_Donchian_Breakout_ATRFilter"
timeframe = "4h"
leverage = 1.0