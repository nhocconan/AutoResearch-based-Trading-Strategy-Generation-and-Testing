#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly trend filter and volume confirmation
# Long when: price breaks above 6h Donchian upper(20), weekly close > weekly EMA50, and volume > 1.5x 20-period average
# Short when: price breaks below 6h Donchian lower(20), weekly close < weekly EMA50, and volume > 1.5x 20-period average
# Exit when price returns to the opposite Donchian level (mean reversion) or opposite breakout
# Uses weekly trend for structural bias (works in both bull/bear via trend alignment) and volume to filter weak breakouts.
# Timeframe: 6h, HTF: 1w. Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_Donchian20_Breakout_1wEMA50_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Calculate volume confirmation on 6h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Get 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50 trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Also get weekly close aligned for trend comparison
    close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
    
    # Calculate 6h Donchian channels (20-period)
    if len(high) >= 20 and len(low) >= 20:
        # Upper channel: highest high of past 20 periods (including current)
        upper_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
        # Lower channel: lowest low of past 20 periods (including current)
        lower_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    else:
        upper_20 = np.full(n, np.nan)
        lower_20 = np.full(n, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(close_1w_aligned[i]) or 
            np.isnan(upper_20[i]) or 
            np.isnan(lower_20[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above upper Donchian, weekly trend up, volume filter
            if (close[i] > upper_20[i] and 
                open_price[i] <= upper_20[i] and  # Ensure breakout happens on this bar
                close_1w_aligned[i] > ema_50_1w_aligned[i] and  # Weekly close > EMA50 (uptrend)
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below lower Donchian, weekly trend down, volume filter
            elif (close[i] < lower_20[i] and 
                  open_price[i] >= lower_20[i] and  # Ensure breakdown happens on this bar
                  close_1w_aligned[i] < ema_50_1w_aligned[i] and  # Weekly close < EMA50 (downtrend)
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns below lower Donchian (mean reversion) or breaks below upper Donchian (failure)
            if close[i] < lower_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above upper Donchian (mean reversion) or breaks above lower Donchian (failure)
            if close[i] > upper_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals