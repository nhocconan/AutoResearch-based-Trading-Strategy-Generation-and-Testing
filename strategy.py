#!/usr/bin/env python3
"""
1h_mtf_ema_crossover_volume_filter
Hypothesis: Use 1d EMA50 for trend direction and 4h EMA20 for momentum, with volume confirmation on 1h. 
Long when price > 1d EMA50, 1h EMA20 crosses above EMA50, and volume > 1.5x average.
Short when price < 1d EMA50, 1h EMA20 crosses below EMA50, and volume > 1.5x average.
Designed for ~20-40 trades/year on 1h with strict multi-timeframe alignment and volume filter to avoid overtrading.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_mtf_ema_crossover_volume_filter"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 4h EMA20 and EMA50 for momentum and crossover
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    ema20_4h = pd.Series(df_4h['close'].values).ewm(span=20, adjust=False).mean().values
    ema50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False).mean().values
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # 1h EMA20 for entry timing
    ema20_1h = pd.Series(close).ewm(span=20, adjust=False).mean().values
    
    # Volume confirmation: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if data not available
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(ema20_4h_aligned[i]) or np.isnan(ema50_4h_aligned[i]) or
            np.isnan(ema20_1h[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average (spike)
        vol_spike = volume[i] > (vol_ma[i] * 1.5)
        
        # EMA crossovers
        ema20_1h_above_ema50_1h = ema20_1h[i] > ema50_1h[i] if i >= 50 else False
        ema20_1h_below_ema50_1h = ema20_1h[i] < ema50_1h[i] if i >= 50 else False
        
        # Calculate 1h EMA50 for crossover detection
        ema50_1h = pd.Series(close).ewm(span=50, adjust=False).mean().values
        
        if position == 1:  # Long position
            # Exit: trend turns bearish or momentum breaks
            if close[i] < ema50_1d_aligned[i] or ema20_1h[i] < ema50_1h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: trend turns bullish or momentum breaks
            if close[i] > ema50_1d_aligned[i] or ema20_1h[i] > ema50_1h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Long: price above 1d EMA50, 1h EMA20 crosses above EMA50, volume spike
            if (close[i] > ema50_1d_aligned[i] and 
                ema20_1h[i] > ema50_1h[i] and 
                ema20_1h[i-1] <= ema50_1h[i-1] and 
                vol_spike):
                position = 1
                signals[i] = 0.20
            # Short: price below 1d EMA50, 1h EMA20 crosses below EMA50, volume spike
            elif (close[i] < ema50_1d_aligned[i] and 
                  ema20_1h[i] < ema50_1h[i] and 
                  ema20_1h[i-1] >= ema50_1h[i-1] and 
                  vol_spike):
                position = -1
                signals[i] = -0.20
    
    return signals