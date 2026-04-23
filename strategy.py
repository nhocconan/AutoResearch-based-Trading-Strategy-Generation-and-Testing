#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation.
Long when price breaks above 12h Donchian upper band (20-period high) AND price > 1d EMA34 (uptrend) AND volume > 2.0x average.
Short when price breaks below 12h Donchian lower band (20-period low) AND price < 1d EMA34 (downtrend) AND volume > 2.0x average.
Exit when price reverts to 12h Donchian middle band (median of upper/lower) or trend reverses (price crosses 1d EMA34).
Uses 12h timeframe to minimize overtrading (target: 50-150 trades over 4 years) and 1d EMA34 for smooth trend filter.
Volume spike ensures high-conviction breakouts. Works in bull markets via breakouts and in bear via short signals.
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
    
    # Load 12h data for Donchian channels - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    vol_12h = df_12h['volume'].values
    
    # Calculate 12h Donchian channels (20-period)
    # Upper band = rolling max of high, Lower band = rolling min of low
    upper_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lower_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    middle_12h = (upper_12h + lower_12h) / 2.0  # Median line
    
    # Load 1d data for EMA34 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to 12h timeframe
    upper_12h_aligned = align_htf_to_ltf(prices, df_12h, upper_12h)
    lower_12h_aligned = align_htf_to_ltf(prices, df_12h, lower_12h)
    middle_12h_aligned = align_htf_to_ltf(prices, df_12h, middle_12h)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    vol_12h_ma = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_12h_ma_aligned = align_htf_to_ltf(prices, df_12h, vol_12h_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(upper_12h_aligned[i]) or np.isnan(lower_12h_aligned[i]) or np.isnan(middle_12h_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_12h_ma_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        upper_val = upper_12h_aligned[i]
        lower_val = lower_12h_aligned[i]
        middle_val = middle_12h_aligned[i]
        ema34_val = ema34_1d_aligned[i]
        vol_ma_val = vol_12h_ma_aligned[i]
        
        # Get current 12h-aligned price and volume
        price = close[i]
        vol_current = volume[i]
        vol_ma_primary = pd.Series(volume).rolling(window=20, min_periods=20).mean().values[i]
        
        if position == 0:
            # Long: price breaks above 12h Donchian upper band AND price > 1d EMA34 (uptrend) AND volume spike
            if (price > upper_val and price > ema34_val and vol_current > 2.0 * vol_ma_primary):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 12h Donchian lower band AND price < 1d EMA34 (downtrend) AND volume spike
            elif (price < lower_val and price < ema34_val and vol_current > 2.0 * vol_ma_primary):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price reverts to 12h Donchian middle band OR price breaks below 1d EMA34 (trend reversal)
                if price <= middle_val or price < ema34_val:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price reverts to 12h Donchian middle band OR price breaks above 1d EMA34 (trend reversal)
                if price >= middle_val or price > ema34_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Donchian20_1dEMA34_Volume"
timeframe = "12h"
leverage = 1.0