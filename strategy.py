#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 12h trend filter and volume confirmation.
# Long: price breaks above Camarilla H3 + price above 12h EMA50 + volume > 1.8x avg volume
# Short: price breaks below Camarilla L3 + price below 12h EMA50 + volume > 1.8x avg volume
# Camarilla levels calculated from 1d data: H3 = close + 1.1*(high-low), L3 = close - 1.1*(high-low)
# Trend filter: only take longs when price > EMA50, shorts when price < EMA50
# Volume confirmation reduces false breakouts
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# Works in both bull and bear markets by using 12h EMA as trend filter

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-day data for Camarilla pivot
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels (using prior day's data)
    H3 = np.full(len(close_1d), np.nan)
    L3 = np.full(len(close_1d), np.nan)
    for i in range(1, len(close_1d)):
        prev_range = high_1d[i-1] - low_1d[i-1]
        H3[i] = close_1d[i-1] + 1.1 * prev_range
        L3[i] = close_1d[i-1] - 1.1 * prev_range
    
    # Align 1d Camarilla to 4h timeframe
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    
    # 12-hour EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 5:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Average volume (20-period = 20*4h = 10 days) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        h3 = H3_aligned[i]
        l3 = L3_aligned[i]
        ema_trend = ema_50_12h_aligned[i]
        
        # Volume confirmation: current volume > 1.8x average volume
        volume_confirm = vol > 1.8 * avg_vol
        
        if position == 0:
            # Long: break above Camarilla H3 + above EMA50 + volume confirmation
            if (price > h3 and 
                price > ema_trend and
                volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: break below Camarilla L3 + below EMA50 + volume confirmation
            elif (price < l3 and 
                  price < ema_trend and
                  volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below Camarilla L3 or below EMA50
            if (price < l3 or
                price < ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above Camarilla H3 or above EMA50
            if (price > h3 or
                price > ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_12h_Camarilla_EMA_Volume"
timeframe = "4h"
leverage = 1.0