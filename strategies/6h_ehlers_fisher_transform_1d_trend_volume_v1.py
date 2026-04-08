#!/usr/bin/env python3
"""
6h_ehlers_fisher_transform_1d_trend_volume_v1
Hypothesis: Ehlers Fisher Transform on 1d timeframe identifies extreme turning points in trend, 
while 6s timeframe provides entry timing with volume confirmation. Works in both bull and bear 
markets by capturing reversals at extremes. Target: 15-35 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ehlers_fisher_transform_1d_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for Fisher Transform
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate Ehlers Fisher Transform (price normalized to [-1, 1] range)
    hl2 = (df_1d['high'] + df_1d['low']) / 2
    # Normalize price to [-1, 1] using 10-period min/max
    min_val = hl2.rolling(window=10, min_periods=10).min()
    max_val = hl2.rolling(window=10, min_periods=10).max()
    # Avoid division by zero
    range_val = max_val - min_val
    price_norm = np.where(range_val > 0, 2 * ((hl2 - min_val) / range_val) - 1, 0)
    # Smooth with 4-period EMA
    price_smoothed = pd.Series(price_norm).ewm(span=4, adjust=False).mean()
    # Fisher Transform
    fish = 0.5 * np.log((1 + price_smoothed) / (1 - price_smoothed + 1e-10))
    fish = np.clip(fish, -0.999, 0.999)  # Prevent extreme values
    # Signal line (3-period EMA of Fisher)
    fish_signal = pd.Series(fish).ewm(span=3, adjust=False).mean()
    
    # Daily EMA for trend filter (50-period)
    ema_50 = df_1d['close'].ewm(span=50, adjust=False).mean()
    
    # Align all daily data to 6h timeframe
    fish_aligned = align_htf_to_ltf(prices, df_1d, fish.values)
    fish_signal_aligned = align_htf_to_ltf(prices, df_1d, fish_signal.values)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50.values)
    
    # Volume confirmation (20-period average = 5 days on 6h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(fish_aligned[i]) or np.isnan(fish_signal_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x average volume
        vol_confirm = volume[i] > 1.3 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: Fisher crosses below signal line or trend turns bearish
            if fish_aligned[i] < fish_signal_aligned[i] or close[i] < ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: Fisher crosses above signal line or trend turns bullish
            if fish_aligned[i] > fish_signal_aligned[i] or close[i] > ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Fisher crosses above signal line with volume and bullish trend
            if (fish_aligned[i] > fish_signal_aligned[i] and vol_confirm and 
                close[i] > ema_50_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Fisher crosses below signal line with volume and bearish trend
            elif (fish_aligned[i] < fish_signal_aligned[i] and vol_confirm and 
                  close[i] < ema_50_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals