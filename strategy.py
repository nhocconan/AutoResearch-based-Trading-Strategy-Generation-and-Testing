#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_breakout_v16"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous 1d bar data (use .shift(1) to avoid look-ahead)
    high_1d_prev = df_1d['high'].shift(1).values
    low_1d_prev = df_1d['low'].shift(1).values
    close_1d_prev = df_1d['close'].shift(1).values
    
    # Calculate 1d Camarilla H4/L4 levels
    pivot_prev = (high_1d_prev + low_1d_prev + close_1d_prev) / 3.0
    range_1d_prev = high_1d_prev - low_1d_prev
    h4_prev = pivot_prev + (range_1d_prev * 1.1 / 2)
    l4_prev = pivot_prev - (range_1d_prev * 1.1 / 2)
    
    # Align to 4h timeframe
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4_prev)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4_prev)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    # Additional filter: avoid low volatility periods using ATR ratio
    atr_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    atr_ma = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    vol_ratio = atr / atr_ma
    vol_filter = vol_ratio > 0.8  # Avoid extremely low volatility
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(200, n):
        if (np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or 
            np.isnan(vol_confirm[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Long: break above H4 with volume confirmation and vol filter
        long_signal = close[i] > h4_aligned[i] and vol_confirm[i] and vol_filter[i]
        # Short: break below L4 with volume confirmation and vol filter
        short_signal = close[i] < l4_aligned[i] and vol_confirm[i] and vol_filter[i]
        
        # Exit when price returns to pivot level
        pivot_prev_val = (high_1d_prev + low_1d_prev + close_1d_prev) / 3.0
        pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_prev_val)
        exit_long = close[i] < pivot_aligned[i]
        exit_short = close[i] > pivot_aligned[i]
        
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals