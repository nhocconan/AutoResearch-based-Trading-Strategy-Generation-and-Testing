#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with 1d Bollinger Bands and volume confirmation.
# Long: Price crosses above upper Bollinger Band + volume > 1.5x average volume.
# Short: Price crosses below lower Bollinger Band + volume > 1.5x average volume.
# Bollinger Bands from 1d provide volatility-based support/resistance structure.
# Volume confirmation ensures breakouts are supported by participation.
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate Bollinger Bands (20, 2) using previous day's data
    upper_bb = np.full(len(close_1d), np.nan)
    lower_bb = np.full(len(close_1d), np.nan)
    for i in range(20, len(close_1d)):
        # Previous 20 days' close
        window = close_1d[i-20:i]
        ma = np.mean(window)
        std = np.std(window)
        upper_bb[i] = ma + 2 * std
        lower_bb[i] = ma - 2 * std
    
    # Average volume (20-period) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    # Align 1d Bollinger Bands to 6h
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(upper_bb_aligned[i]) or np.isnan(lower_bb_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ub = upper_bb_aligned[i]
        lb = lower_bb_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        if position == 0:
            # Long: price crosses above upper BB + volume confirmation
            if (price > ub and volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: price crosses below lower BB + volume confirmation
            elif (price < lb and volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price closes below lower BB (opposite band)
            if price < lb:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price closes above upper BB (opposite band)
            if price > ub:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1d_Bollinger_Bands_Volume"
timeframe = "6h"
leverage = 1.0