#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Wick_Reversal_Volume_Filter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d: Wick rejection signal ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    open_1d = df_1d['open'].values
    close_1d = df_1d['close'].values
    
    # Bullish rejection: long lower wick > 2x body, close near high
    body_1d = np.abs(close_1d - open_1d)
    lower_wick_1d = np.minimum(open_1d, close_1d) - low_1d
    upper_wick_1d = high_1d - np.maximum(open_1d, close_1d)
    
    bullish_wick = (lower_wick_1d > 2 * body_1d) & (close_1d > open_1d)
    bearish_wick = (upper_wick_1d > 2 * body_1d) & (close_1d < open_1d)
    
    bullish_wick_aligned = align_htf_to_ltf(prices, df_1d, bullish_wick.astype(float))
    bearish_wick_aligned = align_htf_to_ltf(prices, df_1d, bearish_wick.astype(float))
    
    # === 4h: Price and volume ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume spike: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get values
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        bullish_wick_val = bullish_wick_aligned[i]
        bearish_wick_val = bearish_wick_aligned[i]
        vol_spike_val = vol_spike[i]
        
        # Skip if any value is NaN
        if np.isnan(bullish_wick_val) or np.isnan(bearish_wick_val) or np.isnan(vol_spike_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: bullish 1d wick rejection + volume spike
            if bullish_wick_val > 0.5 and vol_spike_val:
                signals[i] = 0.25
                position = 1
            # Short: bearish 1d wick rejection + volume spike
            elif bearish_wick_val > 0.5 and vol_spike_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: bearish wick rejection or loss of momentum
            if bearish_wick_val > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: bullish wick rejection or loss of momentum
            if bullish_wick_val > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals