#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_RollingRangeBreakout_VolumeConfirm"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data once before loop
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-day rolling high/low from previous days (avoid look-ahead)
    # Use previous day's data to calculate the range
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    
    # Calculate rolling max/min over previous 20 days
    rolling_high_20 = pd.Series(prev_high_1d).rolling(window=20, min_periods=20).max().values
    rolling_low_20 = pd.Series(prev_low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align to 6h timeframe
    rolling_high_20_6h = align_htf_to_ltf(prices, df_1d, rolling_high_20)
    rolling_low_20_6h = align_htf_to_ltf(prices, df_1d, rolling_low_20)
    
    # Volume confirmation: current 6h volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20
    
    for i in range(start_idx, n):
        if np.isnan(rolling_high_20_6h[i]) or np.isnan(rolling_low_20_6h[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume spike: current volume > 1.5x average
        volume_spike = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long: Price breaks above 20-day rolling high with volume confirmation
            if price > rolling_high_20_6h[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below 20-day rolling low with volume confirmation
            elif price < rolling_low_20_6h[i] and volume_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price returns below 20-day rolling low (reversal signal)
            if price < rolling_low_20_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price returns above 20-day rolling high (reversal signal)
            if price > rolling_high_20_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals