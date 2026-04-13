#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d trend filter and volume confirmation.
# Long: price breaks above H4 resistance + price above 1d EMA(50) + volume > 1.5x avg volume
# Short: price breaks below L4 support + price below 1d EMA(50) + volume > 1.5x avg volume
# Camarilla levels from 1d: H4 = close + 1.1/2*(high-low), L4 = close - 1.1/2*(high-low)
# EMA(50) trend filter avoids counter-trend trades
# Volume confirmation reduces false breakouts
# Target: 50-150 total trades over 4 years (12-37/year) for 4h timeframe
# Works in both bull and bear markets by using 1d EMA as trend filter

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-day data for Camarilla and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels (H4, L4) using prior day's data
    h4 = np.full(len(close_1d), np.nan)
    l4 = np.full(len(close_1d), np.nan)
    for i in range(1, len(close_1d)):
        h4[i] = close_1d[i-1] + 1.1/2 * (high_1d[i-1] - low_1d[i-1])
        l4[i] = close_1d[i-1] - 1.1/2 * (high_1d[i-1] - low_1d[i-1])
    
    # Align 1d Camarilla to 4h timeframe
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    
    # 1d EMA(50) trend filter
    ema_50 = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_50[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_50[i] = (close_1d[i] * 2/51) + (ema_50[i-1] * 49/51)
    
    # Align 1d EMA to 4h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Average volume (20-period = 20*4h = 80h ~ 3.3 days) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        h4_val = h4_aligned[i]
        l4_val = l4_aligned[i]
        ema_val = ema_50_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        if position == 0:
            # Long: break above H4 + above EMA(50) + volume confirmation
            if (price > h4_val and 
                price > ema_val and
                volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: break below L4 + below EMA(50) + volume confirmation
            elif (price < l4_val and 
                  price < ema_val and
                  volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below L4 or below EMA(50)
            if (price < l4_val or
                price < ema_val):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above H4 or above EMA(50)
            if (price > h4_val or
                price > ema_val):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_Camarilla_EMA_Volume"
timeframe = "4h"
leverage = 1.0