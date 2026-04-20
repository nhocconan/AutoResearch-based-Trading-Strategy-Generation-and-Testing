#!/usr/bin/env python3
"""
Strategy: 6h_1d_Pivot_R2S2_Breakout_Volume
Hypothesis: Breakouts beyond daily Camarilla R2/S2 levels with volume confirmation
capture institutional breakout moves. Works in bull/bear markets because:
- In bull markets: breaks above R2 indicate continuation
- In bear markets: breaks below S2 indicate continuation  
- Volume filter ensures breakouts have conviction
- Targets 15-30 trades/year by requiring both level break AND 1.8x volume spike
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels on daily data
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    R2 = pivot_1d + (range_1d * 1.1 / 6)
    S2 = pivot_1d - (range_1d * 1.1 / 6)
    
    # Align to 6h timeframe
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    R2_aligned = align_htf_to_ltf(prices, df_1d, R2)
    S2_aligned = align_htf_to_ltf(prices, df_1d, S2)
    
    # 6h data for entry timing and volume
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # Volume spike detection (20-period)
    vol_ma_20 = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_20)
    
    # ATR for volatility filter (14-period on 6h)
    high_low = high_6h - low_6h
    high_close = np.abs(high_6h - np.roll(close_6h, 1))
    low_close = np.abs(low_6h - np.roll(close_6h, 1))
    high_low[0] = high_6h[0] - low_6h[0]
    high_close[0] = np.abs(high_6h[0] - close_6h[0])
    low_close[0] = np.abs(low_6h[0] - close_6h[0])
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    tr[0] = high_low[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_6h, atr_14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in critical values
        if (np.isnan(R2_aligned[i]) or np.isnan(S2_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i]) or np.isnan(atr_14_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_6h[i]
        vol = volume_6h[i]
        
        if position == 0:
            # Long: price breaks above R2 with volume confirmation
            if (price > R2_aligned[i] and 
                vol > 1.8 * vol_ma_20_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S2 with volume confirmation
            elif (price < S2_aligned[i] and 
                  vol > 1.8 * vol_ma_20_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls back below R2 or ATR-based stop
            if (price < R2_aligned[i] or 
                price < high_6h[i] - 2.0 * atr_14_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises back above S2 or ATR-based stop
            if (price > S2_aligned[i] or 
                price > low_6h[i] + 2.0 * atr_14_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1d_Pivot_R2S2_Breakout_Volume"
timeframe = "6h"
leverage = 1.0