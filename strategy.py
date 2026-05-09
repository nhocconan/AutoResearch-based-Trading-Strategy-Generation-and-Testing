#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_TurtleTrader_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily trend: EMA40 on 1d (slower trend filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    ema40_1d = pd.Series(df_1d['close'].values).ewm(span=40, adjust=False, min_periods=40).mean().values
    ema40_1d_aligned = align_htf_to_ltf(prices, df_1d, ema40_1d)
    
    # Volume filter: volume > 1.5x 30-period SMA (avoid low-volume noise)
    vol_ma30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_filter = volume > 1.5 * vol_ma30
    
    # Turtle Trading: 20-period Donchian breakout (classic breakout system)
    # Upper band: 20-period high
    # Lower band: 20-period low
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if np.isnan(ema40_1d_aligned[i]) or np.isnan(high_20[i]) or \
           np.isnan(low_20[i]) or np.isnan(vol_ma30[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: breakout above 20-period high with daily uptrend and volume
            if (price > high_20[i] and 
                price > ema40_1d_aligned[i] and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
                continue
            
            # Short: breakdown below 20-period low with daily downtrend and volume
            elif (price < low_20[i] and 
                  price < ema40_1d_aligned[i] and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        elif position == 1:
            # Exit long: price returns to 10-period low (classic Turtle exit)
            low_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values
            if price < low_10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to 10-period high
            high_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values
            if price > high_10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals