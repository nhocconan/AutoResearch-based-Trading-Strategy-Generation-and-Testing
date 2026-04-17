#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d Donchian Channel (20-period) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate highest high and lowest low over 20 days
    highest_high = np.full_like(high_1d, np.nan)
    lowest_low = np.full_like(low_1d, np.nan)
    period = 20
    for i in range(len(high_1d)):
        if i >= period - 1:
            highest_high[i] = np.max(high_1d[i-(period-1):i+1])
            lowest_low[i] = np.min(low_1d[i-(period-1):i+1])
        else:
            highest_high[i] = np.max(high_1d[0:i+1]) if i >= 0 else high_1d[0]
            lowest_low[i] = np.min(low_1d[0:i+1]) if i >= 0 else low_1d[0]
    
    # === 1d EMA(50) for trend filter ===
    ema_50 = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 50:
        ema_50[49] = np.mean(close_1d[:50])  # seed
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_1d)):
            ema_50[i] = alpha * close_1d[i] + (1 - alpha) * ema_50[i-1]
    else:
        for i in range(len(close_1d)):
            ema_50[i] = np.mean(close_1d[:i+1]) if i >= 0 else close_1d[0]
    
    # === Align indicators to daily timeframe ===
    donchian_upper = align_htf_to_ltf(prices, df_1d, highest_high)
    donchian_lower = align_htf_to_ltf(prices, df_1d, lowest_low)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # === 1d Volume confirmation (20-period average) ===
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(len(volume)):
        if i >= 19:
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20[i] = np.mean(volume[0:i+1]) if i >= 0 else volume[0]
    
    # Volume filter: current volume > 1.3x 20-day average
    vol_filter = volume > vol_ma_20 * 1.3
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 200
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or 
            np.isnan(ema_50_aligned[i]) or 
            np.isnan(vol_filter[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above Donchian upper band AND above EMA50 AND volume confirmation
            if (close[i] > donchian_upper[i] and 
                close[i] > ema_50_aligned[i] and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below Donchian lower band AND below EMA50 AND volume confirmation
            elif (close[i] < donchian_lower[i] and 
                  close[i] < ema_50_aligned[i] and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price crosses below Donchian lower band OR below EMA50
            if (close[i] < donchian_lower[i] or close[i] < ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above Donchian upper band OR above EMA50
            if (close[i] > donchian_upper[i] or close[i] > ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_EMA50_VolumeFilter"
timeframe = "1d"
leverage = 1.0