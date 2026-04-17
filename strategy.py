#!/usr/bin/env python3
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
    
    # Get daily data for ATR and close (HTF)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily ATR(14)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align daily ATR to 6h timeframe
    atr_6h = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Calculate 6-period EMA on 6h close
    ema6 = pd.Series(close).ewm(span=6, adjust=False, min_periods=6).mean().values
    
    # Volume filter: current volume > 1.8 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 20  # Need ATR(14) + EMA6 + volume MA20
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(atr_6h[i]) or 
            np.isnan(ema6[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.8x 20-period average
        volume_filter = volume[i] > (1.8 * volume_ma20[i])
        
        if position == 0:
            # Long: price > EMA6 and volatility expansion (ATR rising)
            if (close[i] > ema6[i] and 
                atr_6h[i] > atr_6h[i-1] and 
                volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: price < EMA6 and volatility expansion (ATR rising)
            elif (close[i] < ema6[i] and 
                  atr_6h[i] > atr_6h[i-1] and 
                  volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: volatility contraction (ATR falling) or price crosses below EMA6
            if (atr_6h[i] < atr_6h[i-1] or close[i] < ema6[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: volatility contraction (ATR falling) or price crosses above EMA6
            if (atr_6h[i] < atr_6h[i-1] or close[i] > ema6[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_EMA6_VolatilityBreakout_VolumeFilter"
timeframe = "6h"
leverage = 1.0