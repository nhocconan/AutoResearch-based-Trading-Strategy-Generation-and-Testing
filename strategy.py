#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
# Long when price breaks above 20-period high with 1d EMA50 uptrend and volume > 1.8x average.
# Short when price breaks below 20-period low with 1d EMA50 downtrend and volume > 1.8x average.
# Exit when price crosses 10-period EMA in opposite direction.
# Uses Donchian channels for breakout strength, EMA50 for trend filter, volume for confirmation.
# Designed for low trade frequency (target: 15-30 trades/year) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_period = 50
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= ema_period:
        ema_1d[ema_period - 1] = np.mean(close_1d[:ema_period])
        for i in range(ema_period, len(close_1d)):
            ema_1d[i] = (close_1d[i] * (2 / (ema_period + 1)) + 
                         ema_1d[i - 1] * (1 - (2 / (ema_period + 1))))
    
    # Calculate 10-period EMA for exit
    ema10_period = 10
    ema10 = np.full(n, np.nan)
    if n >= ema10_period:
        ema10[ema10_period - 1] = np.mean(close[:ema10_period])
        for i in range(ema10_period, n):
            ema10[i] = (close[i] * (2 / (ema10_period + 1)) + 
                        ema10[i - 1] * (1 - (2 / (ema10_period + 1))))
    
    # Calculate Donchian channels (20-period)
    donchian_period = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(donchian_period - 1, n):
        upper[i] = np.max(high[i - donchian_period + 1:i + 1])
        lower[i] = np.min(low[i - donchian_period + 1:i + 1])
    
    # Align 1d EMA50 to 12h timeframe
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume MA for confirmation (20-period)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i - 19:i + 1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian(20), EMA50, EMA10, and volume MA20
    start_idx = max(donchian_period, ema_period - 1, ema10_period - 1, 19)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(ema10[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: require volume significantly above average
        vol_filter = vol_now > 1.8 * vol_avg
        
        if position == 0:
            # Long: price breaks above 20-period high with 1d EMA50 uptrend and volume filter
            if (price > upper[i] and 
                price > ema_1d_aligned[i] and vol_filter):
                signals[i] = size
                position = 1
            # Short: price breaks below 20-period low with 1d EMA50 downtrend and volume filter
            elif (price < lower[i] and 
                  price < ema_1d_aligned[i] and vol_filter):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below 10-period EMA
            if price < ema10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above 10-period EMA
            if price > ema10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Donchian20_Breakout_1dEMA50_Volume"
timeframe = "12h"
leverage = 1.0