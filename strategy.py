#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy combining 12h EMA trend filter with 4h Donchian channel breakout and volume confirmation.
# Uses 12h EMA50 for trend direction (avoids whipsaw in sideways markets), 4h Donchian(20) breakouts for entry,
# and volume > 1.5x 20-period average for confirmation. Designed for fewer, high-quality trades to minimize fee drag.
# Works in bull markets via trend-following breakouts and in bear markets via short breakdowns with trend filter.
# Target: 20-40 trades/year (~80-160 over 4 years) to stay under fee drag threshold.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend
    close_12h = df_12h['close'].values
    ema_50_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 50:
        multiplier = 2 / (50 + 1)
        ema_50_12h[49] = np.mean(close_12h[:50])
        for i in range(50, len(close_12h)):
            ema_50_12h[i] = (close_12h[i] * multiplier) + (ema_50_12h[i-1] * (1 - multiplier))
    
    # Align 12h EMA50 to 4h
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 4h Donchian channels (20-period)
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    lookback = 20
    for i in range(lookback - 1, n):
        highest_high[i] = np.max(high[i-lookback+1:i+1])
        lowest_low[i] = np.min(low[i-lookback+1:i+1])
    
    # Calculate 20-period volume average for spike confirmation
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # 25% position size
    
    # Warmup: max of 12h EMA50 (50), Donchian (20), vol MA (20) + buffer
    start_idx = max(50, 20, vol_period) + 5
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        vol_filter = vol_ratio > 1.5  # Volume at least 1.5x average
        
        if position == 0:
            # Long: Uptrend (price > 12h EMA50) + breakout above Donchian high + volume
            if price > ema_50_12h_aligned[i] and price > highest_high[i] and vol_filter:
                signals[i] = size
                position = 1
            # Short: Downtrend (price < 12h EMA50) + breakdown below Donchian low + volume
            elif price < ema_50_12h_aligned[i] and price < lowest_low[i] and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price breaks below Donchian low or trend reverses
            if price < lowest_low[i] or price < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Price breaks above Donchian high or trend reverses
            if price > highest_high[i] or price > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_EMA50_12h_Donchian20_Volume"
timeframe = "4h"
leverage = 1.0