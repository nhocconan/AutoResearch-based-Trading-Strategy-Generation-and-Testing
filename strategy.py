#!/usr/bin/env python3
"""
4h_1d_Trend_With_Volume_Confirmation
Trend-following strategy using 1d EMA34 for direction and 4h Donchian breakout for entry.
Long when price breaks above Donchian(20) high and close > 1d EMA34.
Short when price breaks below Donchian(20) low and close < 1d EMA34.
Exit when price crosses back through Donchian midpoint or trend reverses.
Volume confirmation: require volume > 1.5x 20-period average.
Target: 20-30 trades/year per symbol.
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
    
    # 4h Donchian channel (20-period)
    donch_period = 20
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    
    for i in range(donch_period - 1, n):
        donch_high[i] = np.max(high[i - donch_period + 1:i + 1])
        donch_low[i] = np.min(low[i - donch_period + 1:i + 1])
    
    # Volume average (20-period)
    vol_avg = np.full(n, np.nan)
    for i in range(19, n):
        vol_avg[i] = np.mean(volume[i - 19:i + 1])
    
    # Get 1d data for higher timeframe trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 for trend filter
    ema_period = 34
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= ema_period:
        ema_1d[ema_period - 1] = np.mean(close_1d[:ema_period])
        for i in range(ema_period, len(close_1d)):
            ema_1d[i] = (close_1d[i] * (2 / (ema_period + 1)) + 
                         ema_1d[i - 1] * (1 - (2 / (ema_period + 1))))
    
    # Align 1d EMA34 to 4h timeframe
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian, volume average, and EMA1d
    start_idx = max(donch_period - 1, 19, ema_period - 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(vol_avg[i]) or np.isnan(ema_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_average = vol_avg[i]
        donch_high_val = donch_high[i]
        donch_low_val = donch_low[i]
        ema1d_val = ema_1d_aligned[i]
        
        # Donchian midpoint for exit
        donch_mid = (donch_high_val + donch_low_val) / 2
        
        if position == 0:
            # Long: price breaks above Donchian high, volume > 1.5x avg, and price > 1d EMA34
            if (price > donch_high_val and vol > 1.5 * vol_average and price > ema1d_val):
                signals[i] = size
                position = 1
            # Short: price breaks below Donchian low, volume > 1.5x avg, and price < 1d EMA34
            elif (price < donch_low_val and vol > 1.5 * vol_average and price < ema1d_val):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below Donchian midpoint or trend reverses (price < 1d EMA34)
            if price < donch_mid or price < ema1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above Donchian midpoint or trend reverses (price > 1d EMA34)
            if price > donch_mid or price > ema1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_1d_Trend_With_Volume_Confirmation"
timeframe = "4h"
leverage = 1.0