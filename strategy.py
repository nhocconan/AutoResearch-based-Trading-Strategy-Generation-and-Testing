#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with 1d ATR trend filter and volume confirmation
- Uses 6h Donchian channels (20-period high/low) for breakout detection
- 1d ATR(14) defines volatility regime: only trade when ATR > 20-period SMA (high volatility)
- Volume confirmation (> 1.5x 20-period average) filters low-conviction breakouts
- Designed for 6h timeframe targeting 12-37 trades/year (50-150 over 4 years)
- Works in both bull and bear markets by requiring high volatility regime (avoids chop)
- Donchian breakouts capture momentum; volatility filter ensures trending conditions
"""

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
    
    # Calculate 6h Donchian channels (20-period)
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d ATR(14) for volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 15:  # Need at least 14 for ATR + 1 for SMA
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_sma = pd.Series(atr_14).rolling(window=20, min_periods=20).mean().values
    vol_regime = atr_14 > atr_sma  # High volatility regime
    
    # Align 1d volatility regime to 6h
    vol_regime_aligned = align_htf_to_ltf(prices, df_1d, vol_regime.astype(float))
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 20)  # For Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(high_ma[i]) or np.isnan(low_ma[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(vol_regime_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above 6h Donchian high with high vol and volume
            long_breakout = (close[i] > high_ma[i] and 
                           vol_regime_aligned[i] > 0.5 and  # High volatility regime
                           volume[i] > 1.5 * vol_ma[i])
            
            # Short conditions: price breaks below 6h Donchian low with high vol and volume
            short_breakout = (close[i] < low_ma[i] and 
                            vol_regime_aligned[i] > 0.5 and  # High volatility regime
                            volume[i] > 1.5 * vol_ma[i])
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
            elif short_breakout:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: opposite Donchian breakout or volatility regime ends
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below 6h Donchian low or low volatility
                if (close[i] < low_ma[i] or 
                    vol_regime_aligned[i] <= 0.5):
                    exit_signal = True
            elif position == -1:
                # Exit short: price breaks above 6h Donchian high or low volatility
                if (close[i] > high_ma[i] or 
                    vol_regime_aligned[i] <= 0.5):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Donchian20_Breakout_1dATR_VolumeFilter"
timeframe = "6h"
leverage = 1.0