#!/usr/bin/env python3
name = "6h_Ehlers_Fisher_Transform_Signal"
timeframe = "6h"
leverage = 1.0

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
    
    # Get 1d data for Fisher Transform and regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 12h data for volume filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Ehlers Fisher Transform on 1d closes
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    hl2_1d = (high_1d + low_1d) / 2
    
    # Normalize price to [-1, 1] range over 10-period window
    def normalize_series(series, length):
        highest = pd.Series(series).rolling(window=length, min_periods=length).max().values
        lowest = pd.Series(series).rolling(window=length, min_periods=length).min().values
        # Avoid division by zero
        diff = highest - lowest
        diff = np.where(diff == 0, 1, diff)
        return 2 * ((series - lowest) / diff) - 1
    
    price_norm = normalize_series(hl2_1d, 10)
    
    # Fisher Transform: 0.5 * ln((1+price)/(1-price))
    # Clamp to avoid division by zero or log of negative
    price_norm_clamped = np.clip(price_norm, -0.999, 0.999)
    fish = 0.5 * np.log((1 + price_norm_clamped) / (1 - price_norm_clamped))
    
    # Smoothed Fisher (signal line)
    fish_smooth = pd.Series(fish).ewm(span=3, adjust=False).mean().values
    
    # Volume filter: 12h volume > 1.5 * 20-period average
    vol_12h = df_12h['volume'].values
    vol_avg_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_filter_12h = vol_12h > (vol_avg_12h * 1.5)
    
    # Align all to 6h timeframe
    fish_aligned = align_htf_to_ltf(prices, df_1d, fish)
    fish_smooth_aligned = align_htf_to_ltf(prices, df_1d, fish_smooth)
    vol_filter_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_filter_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # enough for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(fish_aligned[i]) or np.isnan(fish_smooth_aligned[i]) or 
            np.isnan(vol_filter_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Fisher crosses above -1.5 (oversold reversal)
            if fish[i] > -1.5 and fish_smooth[i] <= -1.5 and vol_filter_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Fisher crosses below +1.5 (overbought reversal)
            elif fish[i] < 1.5 and fish_smooth[i] >= 1.5 and vol_filter_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: Fisher crosses the signal line in opposite direction
            if position == 1:
                if fish[i] < fish_smooth[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if fish[i] > fish_smooth[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals