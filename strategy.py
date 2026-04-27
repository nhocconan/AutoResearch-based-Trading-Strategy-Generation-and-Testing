#!/usr/bin/env python3
"""
1d Donchian(20) Breakout with 1w EMA50 Trend Filter and Volume Confirmation
Long when price breaks above Donchian upper channel with 1w EMA50 uptrend and volume > 1.5x average.
Short when price breaks below Donchian lower channel with 1w EMA50 downtrend and volume > 1.5x average.
Exit when price crosses opposite Donchian band or trend filter fails.
Designed for low-frequency, high-conviction trades targeting 7-25 trades/year.
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
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50 for trend filter
    ema_period = 50
    ema_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= ema_period:
        ema_1w[ema_period - 1] = np.mean(close_1w[:ema_period])
        for i in range(ema_period, len(close_1w)):
            ema_1w[i] = (close_1w[i] * (2 / (ema_period + 1)) + 
                         ema_1w[i - 1] * (1 - (2 / (ema_period + 1))))
    
    # Calculate Donchian Channel (20-period)
    donch_len = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(donch_len - 1, n):
        upper[i] = np.max(high[i - donch_len + 1:i + 1])
        lower[i] = np.min(low[i - donch_len + 1:i + 1])
    
    # Calculate 20-period volume MA for confirmation
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i - 19:i + 1])
    
    # Align 1w EMA50 to daily timeframe
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian, EMA50, and volume MA20
    start_idx = max(donch_len - 1, ema_period - 1, 19)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(ema_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: require volume above average
        vol_filter = vol_now > 1.5 * vol_avg
        
        if position == 0:
            # Long: price breaks above Donchian upper with 1w EMA50 uptrend and volume filter
            if (price > upper[i] and 
                price > ema_1w_aligned[i] and vol_filter):
                signals[i] = size
                position = 1
            # Short: price breaks below Donchian lower with 1w EMA50 downtrend and volume filter
            elif (price < lower[i] and 
                  price < ema_1w_aligned[i] and vol_filter):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below Donchian lower OR trend fails
            if (price < lower[i] or price < ema_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above Donchian upper OR trend fails
            if (price > upper[i] or price > ema_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Donchian20_Breakout_1wEMA50_Volume"
timeframe = "1d"
leverage = 1.0