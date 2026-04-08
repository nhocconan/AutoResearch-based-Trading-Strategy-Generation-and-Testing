#!/usr/bin/env python3
# 4h_1d_donchian_breakout_v1
# Hypothesis: 4-hour Donchian channel breakout with 1-day trend filter and volume confirmation.
# Long when: price breaks above 20-period Donchian high, price > 1-day EMA50, and volume > 1.5x average volume.
# Short when: price breaks below 20-period Donchian low, price < 1-day EMA50, and volume > 1.5x average volume.
# Exit when price crosses back through the Donchian midpoint or trend fails.
# Uses 1-day EMA for trend filter to avoid counter-trend trades, volume confirmation to ensure breakout strength.
# Target: 20-40 trades/year to minimize fee dust while capturing strong trending moves.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_donchian_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 20-period Donchian channels
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(20, n):
        highest_high[i] = np.max(high[i-19:i+1])
        lowest_low[i] = np.min(low[i-19:i+1])
    
    # Calculate Donchian midpoint for exit
    donchian_mid = (highest_high + lowest_low) / 2
    
    # Calculate average volume (20-period)
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-19:i+1])
    
    # Get 1-day EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_1d_50 = np.full(len(close_1d), np.nan)
    ema_1d_50[49] = np.mean(close_1d[:50])
    for i in range(50, len(close_1d)):
        ema_1d_50[i] = close_1d[i] * 0.0377 + ema_1d_50[i-1] * 0.9623
    
    ema_1d_50_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_50)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(avg_volume[i]) or np.isnan(ema_1d_50_aligned[i]):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long
            # Exit: price below Donchian midpoint OR below 1-day EMA50
            if close[i] < donchian_mid[i] or close[i] < ema_1d_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price above Donchian midpoint OR above 1-day EMA50
            if close[i] > donchian_mid[i] or close[i] > ema_1d_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry: bullish breakout with volume confirmation
            if (close[i] > highest_high[i] and 
                close[i] > ema_1d_50_aligned[i] and 
                volume[i] > 1.5 * avg_volume[i]):
                position = 1
                signals[i] = 0.25
            # Entry: bearish breakout with volume confirmation
            elif (close[i] < lowest_low[i] and 
                  close[i] < ema_1d_50_aligned[i] and 
                  volume[i] > 1.5 * avg_volume[i]):
                position = -1
                signals[i] = -0.25
    
    return signals