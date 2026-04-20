#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data for 12h proxy (using 1d data to avoid data gaps)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 12h equivalent ATR (use 2-period ATR on daily)
    high_low = high_1d - low_1d
    high_close = np.abs(high_1d - np.roll(close_1d, 1))
    low_close = np.abs(low_1d - np.roll(close_1d, 1))
    high_low[0] = high_1d[0] - low_1d[0]
    high_close[0] = np.abs(high_1d[0] - close_1d[0])
    low_close[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr_2d = pd.Series(tr).rolling(window=2, min_periods=2).mean().values
    atr_12h_aligned = align_htf_to_ltf(prices, df_1d, atr_2d)
    
    # Calculate 12h equivalent volume average (2-period)
    vol_ma_12h = pd.Series(volume_1d).rolling(window=2, min_periods=2).mean().values
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_12h)
    
    # Calculate 12h equivalent price range (high-low over 2 days)
    high_2d = pd.Series(high_1d).rolling(window=2, min_periods=2).max().values
    low_2d = pd.Series(low_1d).rolling(window=2, min_periods=2).min().values
    high_2d_aligned = align_htf_to_ltf(prices, df_1d, high_2d)
    low_2d_aligned = align_htf_to_ltf(prices, df_1d, low_2d)
    
    # Calculate range position (0 = at low, 1 = at high)
    range_size = high_2d_aligned - low_2d_aligned
    range_position = np.where(range_size > 0, (close_1d - low_2d_aligned) / range_size, 0.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in critical values
        if (np.isnan(atr_12h_aligned[i]) or np.isnan(vol_ma_12h_aligned[i]) or 
            np.isnan(range_position[i]) or np.isnan(close_1d[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_1d[i]
        vol = volume_1d[i]
        rpos = range_position[i]
        atr = atr_12h_aligned[i]
        vol_ma = vol_ma_12h_aligned[i]
        
        if position == 0:
            # Long: price in lower 30% of range with volume expansion and sufficient volatility
            if (rpos < 0.3 and 
                vol > 1.5 * vol_ma and 
                atr > 0):
                signals[i] = 0.25
                position = 1
            # Short: price in upper 70% of range with volume expansion and sufficient volatility
            elif (rpos > 0.7 and 
                  vol > 1.5 * vol_ma and 
                  atr > 0):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price moves to middle/upper range or volume drops
            if rpos > 0.5 or vol < 0.7 * vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price moves to middle/lower range or volume drops
            if rpos < 0.5 or vol < 0.7 * vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_RangePosition_VolumeFilter"
timeframe = "4h"
leverage = 1.0