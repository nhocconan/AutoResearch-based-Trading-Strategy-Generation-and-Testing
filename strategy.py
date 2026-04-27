#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d trend filter and volume confirmation.
# Long when price breaks above Donchian(20) high and 1d EMA(50) is rising.
# Short when price breaks below Donchian(20) low and 1d EMA(50) is falling.
# Uses volume > 1.5x 20-period average for confirmation.
# Exit on opposite Donchian break or when 1d EMA direction changes.
# Designed to work in both bull (breakouts) and bear (breakdowns) markets.
# Target: 20-50 trades/year to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
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
    
    # Calculate 1d EMA(50) for trend direction
    ema_50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_50_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = (close_1d[i] * 0.04) + (ema_50_1d[i-1] * 0.96)
    
    # Calculate 1d EMA direction (rising/falling)
    ema_rising_1d = np.full(len(close_1d), False)
    ema_falling_1d = np.full(len(close_1d), False)
    for i in range(1, len(ema_50_1d)):
        if not np.isnan(ema_50_1d[i]) and not np.isnan(ema_50_1d[i-1]):
            ema_rising_1d[i] = ema_50_1d[i] > ema_50_1d[i-1]
            ema_falling_1d[i] = ema_50_1d[i] < ema_50_1d[i-1]
    
    # Align 1d EMA direction to 4h timeframe
    ema_rising_aligned = align_htf_to_ltf(prices, df_1d, ema_rising_1d.astype(float))
    ema_falling_aligned = align_htf_to_ltf(prices, df_1d, ema_falling_1d.astype(float))
    
    # Calculate Donchian channels (20-period)
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    for i in range(19, n):
        donch_high[i] = np.max(high[i-19:i+1])
        donch_low[i] = np.min(low[i-19:i+1])
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    vol_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need 20-period Donchian, 20-period volume MA, and 50-period EMA
    start_idx = max(19, 19, 49)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(ema_rising_aligned[i]) or 
            np.isnan(ema_falling_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ok = vol_filter[i]
        
        if position == 0:
            # Long: price breaks above Donchian high, 1d EMA rising, volume confirmation
            if price > donch_high[i] and ema_rising_aligned[i] > 0.5 and vol_ok:
                signals[i] = size
                position = 1
            # Short: price breaks below Donchian low, 1d EMA falling, volume confirmation
            elif price < donch_low[i] and ema_falling_aligned[i] > 0.5 and vol_ok:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below Donchian low OR 1d EMA starts falling
            if price < donch_low[i] or ema_falling_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price breaks above Donchian high OR 1d EMA starts rising
            if price > donch_high[i] or ema_rising_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian_Breakout_1dEMA50_Volume"
timeframe = "4h"
leverage = 1.0