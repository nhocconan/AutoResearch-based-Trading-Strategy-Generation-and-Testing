#!/usr/bin/env python3
"""
6h_HTF_12h_Supertrend_VolumeBreakout_V1
Hypothesis: 6h breakouts above/below 12h Supertrend with volume confirmation (>1.5x 20-period volume MA). 
Uses 12h HTF for Supertrend (ATR=10, mult=3.0) to define trend direction and filter false breakouts. 
Works in bull/bear markets: Supertrend adapts to volatility, volume confirms institutional interest. 
Target: 12-37 trades/year (50-150 total over 4 years) on BTC/ETH/SOL. Discrete sizing (0.25) to minimize fee churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (12h for Supertrend)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # === 12h Supertrend (ATR=10, mult=3.0) ===
    # Calculate ATR
    tr1 = pd.Series(high_12h[1:]) - pd.Series(low_12h[1:])
    tr2 = abs(pd.Series(high_12h[1:]) - pd.Series(close_12h[:-1]))
    tr3 = abs(pd.Series(low_12h[1:]) - pd.Series(close_12h[:-1]))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Supertrend calculation
    hl2 = (high_12h + low_12h) / 2.0
    upper = hl2 + (3.0 * atr)
    lower = hl2 - (3.0 * atr)
    
    supertrend = np.zeros_like(close_12h)
    direction = np.ones_like(close_12h)  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = hl2[0]
    direction[0] = 1
    
    for i in range(1, len(close_12h)):
        if close_12h[i] > supertrend[i-1]:
            supertrend[i] = max(lower[i], supertrend[i-1])
            direction[i] = 1
        else:
            supertrend[i] = min(upper[i], supertrend[i-1])
            direction[i] = -1
    
    # Align Supertrend and direction to 6h timeframe
    supertrend_aligned = align_htf_to_ltf(prices, df_12h, supertrend)
    direction_aligned = align_htf_to_ltf(prices, df_12h, direction)
    
    # === 6h Indicators (primary timeframe) ===
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # Volume MA (20-period) for spike detection
    vol_ma = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if indicators not ready
        if (np.isnan(supertrend_aligned[i]) or np.isnan(direction_aligned[i]) 
            or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_6h[i]
        vol = volume_6h[i]
        vol_ok = vol > 1.5 * vol_ma[i]  # volume spike confirmation
        
        if position == 0:
            # Long: price breaks above Supertrend + volume spike + uptrend (direction=1)
            if price > supertrend_aligned[i] and vol_ok and direction_aligned[i] == 1:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Supertrend + volume spike + downtrend (direction=-1)
            elif price < supertrend_aligned[i] and vol_ok and direction_aligned[i] == -1:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below Supertrend
            if price < supertrend_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above Supertrend
            if price > supertrend_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_HTF_12h_Supertrend_VolumeBreakout_V1"
timeframe = "6h"
leverage = 1.0