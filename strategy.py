#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(15) breakout with 1d EMA(50) trend filter and volume confirmation.
# Long: price breaks above Donchian high + price > 1d EMA(50) + volume > 1.5x avg volume
# Short: price breaks below Donchian low + price < 1d EMA(50) + volume > 1.5x avg volume
# Uses daily EMA for trend filter to avoid whipsaws in sideways markets
# Volume confirmation reduces false breakouts
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# Works in both bull and bear markets by using daily EMA as trend filter

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-day data for EMA(50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # EMA(50) on 1d timeframe
    ema_50_1d = np.full(len(close_1d), np.nan)
    alpha = 2.0 / (50 + 1)
    for i in range(len(close_1d)):
        if i == 0:
            ema_50_1d[i] = close_1d[i]
        elif np.isnan(ema_50_1d[i-1]):
            ema_50_1d[i] = close_1d[i]
        else:
            ema_50_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema_50_1d[i-1]
    
    # Align 1d EMA to 12h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Donchian(15) on 12h timeframe
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    for i in range(15, n):
        donch_high[i] = np.max(high[i-15:i])
        donch_low[i] = np.min(low[i-15:i])
    
    # Average volume (15-period = 15*12h = 7.5 days) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(15, n):
        avg_volume[i] = np.mean(volume[i-15:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(15, n):
        # Skip if any required data is not ready
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema = ema_50_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        if position == 0:
            # Long: break above Donchian high + above EMA + volume confirmation
            if (price > donch_high[i] and 
                price > ema and
                volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: break below Donchian low + below EMA + volume confirmation
            elif (price < donch_low[i] and 
                  price < ema and
                  volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below Donchian low or below EMA
            if (price < donch_low[i] or
                price < ema):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above Donchian high or above EMA
            if (price > donch_high[i] or
                price > ema):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_Donchian_EMA_Volume"
timeframe = "12h"
leverage = 1.0