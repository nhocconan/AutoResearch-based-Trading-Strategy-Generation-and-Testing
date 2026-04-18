#!/usr/bin/env python3
"""
4h_Pivot_R1_S1_Breakout_Volume_ATRFilter
4h strategy using daily Camarilla pivot levels (R1/S1) for breakout entries with volume confirmation and ATR-based stoploss.
- Long: Close breaks above daily R1 + volume > 1.5x 20-period average + ATR filter
- Short: Close breaks below daily S1 + volume > 1.5x 20-period average + ATR filter
- Exit: Opposite breakout or volatility-based stop
Designed for ~25-40 trades/year per symbol (100-160 total over 4 years)
Works in bull markets (breakout continuation) and bear markets (breakdown continuation)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily Camarilla pivot levels
    # Pivot = (High + Low + Close) / 3
    # R1 = Pivot + (High - Low) * 1.1 / 12
    # S1 = Pivot - (High - Low) * 1.1 / 12
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = pivot_1d + (high_1d - low_1d) * 1.1 / 12.0
    s1_1d = pivot_1d - (high_1d - low_1d) * 1.1 / 12.0
    
    # Align daily pivot levels to 4h
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Volume confirmation: 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for stoploss (14-period)
    tr1 = np.maximum(high - low, np.absolute(high - np.roll(close, 1)))
    tr2 = np.absolute(low - np.roll(close, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high[0] - low[0]  # First period TR
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need enough for volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma_20[i]
        
        # Breakout conditions
        breakout_up = close[i] > r1_aligned[i]
        breakdown_down = close[i] < s1_aligned[i]
        
        if position == 0:
            # Long: volume + breakout above R1
            if vol_confirm and breakout_up:
                signals[i] = 0.25
                position = 1
            # Short: volume + breakdown below S1
            elif vol_confirm and breakdown_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: opposite breakdown or volatility stop
            if breakdown_down or close[i] < prices['high'].max() - 2.0 * atr[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: opposite breakout or volatility stop
            if breakout_up or close[i] > prices['low'].min() + 2.0 * atr[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Pivot_R1_S1_Breakout_Volume_ATRFilter"
timeframe = "4h"
leverage = 1.0