#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation.
Long when price breaks above 4h Donchian upper band (20) AND price > 12h EMA50 (uptrend) AND volume > 1.5x 20-period average.
Short when price breaks below 4h Donchian lower band (20) AND price < 12h EMA50 (downtrend) AND volume > 1.5x 20-period average.
Exit when price reverts to 4h Donchian middle band or trend reverses (price crosses 12h EMA50).
Uses discrete position sizing (0.25) to minimize fee churn. Designed for low trade frequency (~20-40/year) to capture strong breakouts while avoiding false signals.
Works in both bull and bear markets by requiring trend confirmation via 12h EMA50 for breakout entries.
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
    
    # Load 12h data for EMA50 trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate EMA50 for 12h trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate 4h Donchian channels (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    upper_band = high_max
    lower_band = low_min
    middle_band = (upper_band + lower_band) / 2
    
    # Volume average (20-period) on 4h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(upper_band[i]) or 
            np.isnan(lower_band[i]) or np.isnan(middle_band[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_val = ema50_12h_aligned[i]
        upper_val = upper_band[i]
        lower_val = lower_band[i]
        middle_val = middle_band[i]
        vol_ma_val = vol_ma[i]
        vol_current = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: price breaks above upper band AND price > 12h EMA50 (uptrend) AND volume spike
            if (price > upper_val and price > ema50_val and vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower band AND price < 12h EMA50 (downtrend) AND volume spike
            elif (price < lower_val and price < ema50_val and vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price reverts to middle band OR price breaks below 12h EMA50 (trend reversal)
                if price <= middle_val or price < ema50_val:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price reverts to middle band OR price breaks above 12h EMA50 (trend reversal)
                if price >= middle_val or price > ema50_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian20_12hEMA50_Volume_Breakout"
timeframe = "4h"
leverage = 1.0