#!/usr/bin/env python3
"""
4H_Aroon_Trend_Filter_With_Volume_Confirmation
Long when Aroon Up > 70 and Aroon Down < 30 with price above 1D EMA50 and volume > 1.5x average.
Short when Aroon Down > 70 and Aroon Up < 30 with price below 1D EMA50 and volume > 1.5x average.
Exit when Aroon Up < 50 or Aroon Down < 50 (trend weakness).
Aroon measures trend strength and targets 20-40 trades per year for low turnover.
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
    
    # Get 1D data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1D EMA50 for trend filter
    ema_period = 50
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= ema_period:
        ema_1d[ema_period - 1] = np.mean(close_1d[:ema_period])
        for i in range(ema_period, len(close_1d)):
            ema_1d[i] = (close_1d[i] * (2 / (ema_period + 1)) + 
                         ema_1d[i - 1] * (1 - (2 / (ema_period + 1))))
    
    # Calculate Aroon on 4H
    aroon_period = 25
    aroon_up = np.full(n, np.nan)
    aroon_down = np.full(n, np.nan)
    
    for i in range(aroon_period - 1, n):
        # Periods since highest high
        highest_high_idx = np.argmax(high[i - aroon_period + 1:i + 1])
        periods_since_high = aroon_period - 1 - highest_high_idx
        aroon_up[i] = ((aroon_period - periods_since_high) / aroon_period) * 100
        
        # Periods since lowest low
        lowest_low_idx = np.argmin(low[i - aroon_period + 1:i + 1])
        periods_since_low = aroon_period - 1 - lowest_low_idx
        aroon_down[i] = ((aroon_period - periods_since_low) / aroon_period) * 100
    
    # Align 1D EMA to 4H timeframe
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume MA for confirmation (20-period)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i - 19:i + 1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Aroon, EMA1D, and volume MA20
    start_idx = max(aroon_period - 1, ema_period - 1, 19)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(aroon_up[i]) or np.isnan(aroon_down[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: require volume above average
        vol_filter = vol_now > 1.5 * vol_avg
        
        if position == 0:
            # Long: Aroon Up > 70, Aroon Down < 30, price above 1D EMA50, volume filter
            if (aroon_up[i] > 70 and aroon_down[i] < 30 and 
                price > ema_1d_aligned[i] and vol_filter):
                signals[i] = size
                position = 1
            # Short: Aroon Down > 70, Aroon Up < 30, price below 1D EMA50, volume filter
            elif (aroon_down[i] > 70 and aroon_up[i] < 30 and 
                  price < ema_1d_aligned[i] and vol_filter):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Aroon Up < 50 or Aroon Down < 50 (trend weakness)
            if aroon_up[i] < 50 or aroon_down[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: Aroon Down < 50 or Aroon Up < 50 (trend weakness)
            if aroon_down[i] < 50 or aroon_up[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4H_Aroon_Trend_Filter_With_Volume_Confirmation"
timeframe = "4h"
leverage = 1.0