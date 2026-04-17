#!/usr/bin/env python3
"""
4h_GoldenCross_VolumeSurge_v1
Price crosses above/below SMA50 with volume surge confirmation.
Uses 12h EMA200 as long-term trend filter.
Exit on opposite SMA crossover or volume exhaustion.
Designed to capture sustained trends with institutional participation signals.
Target: 60-120 total trades over 4 years (15-30/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === SMA(50) ===
    sma_50 = pd.Series(close).rolling(window=50, min_periods=50).mean().values
    
    # === Volume Surge: volume > 1.5x 20-period average ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_surge = volume > (1.5 * vol_ma_20)
    
    # === 12h EMA200 for trend filter ===
    df_12h = get_htf_data(prices, '12h')
    ema_200_12h = pd.Series(df_12h['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_200_12h)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(sma_50[i]) or 
            np.isnan(ema_200_12h_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price crosses above SMA50, volume surge, above 12h EMA200
            if (close[i] > sma_50[i] and 
                close[i-1] <= sma_50[i-1] and 
                volume_surge[i] and 
                close[i] > ema_200_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price crosses below SMA50, volume surge, below 12h EMA200
            elif (close[i] < sma_50[i] and 
                  close[i-1] >= sma_50[i-1] and 
                  volume_surge[i] and 
                  close[i] < ema_200_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price crosses below SMA50 OR below 12h EMA200
            if (close[i] < sma_50[i] and close[i-1] >= sma_50[i-1]) or \
               (close[i] < ema_200_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above SMA50 OR above 12h EMA200
            if (close[i] > sma_50[i] and close[i-1] <= sma_50[i-1]) or \
               (close[i] > ema_200_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_GoldenCross_VolumeSurge_v1"
timeframe = "4h"
leverage = 1.0