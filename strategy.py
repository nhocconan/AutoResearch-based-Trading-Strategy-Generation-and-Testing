#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d timeframe with 1w ATR-based breakout and volume confirmation.
# Uses 1w ATR to define breakout levels, with 1d close breaking above/below
# previous day's high/low plus ATR. Volume filter ensures breakout strength.
# Works in both bull and bear markets by capturing breakouts in either direction.
# Target: 30-100 total trades over 4 years (7-25/year).
name = "1d_1w_ATRBreakout_VolumeFilter"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for ATR calculation (called ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ATR on 1w timeframe
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period TR is just high-low
    atr_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align ATR to 1d timeframe
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # Volume filter: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for ATR
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(atr_1w_aligned[i]) or np.isnan(volume_ma[i]):
            signals[i] = 0.0
            continue
            
        # Calculate breakout levels using previous day high/low + ATR
        prev_high = high[i-1]
        prev_low = low[i-1]
        atr_val = atr_1w_aligned[i]
        
        long_breakout = prev_high + atr_val
        short_breakout = prev_low - atr_val
        
        if position == 0:
            # Long when price breaks above previous high + ATR with volume
            if close[i] > long_breakout and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below previous low - ATR with volume
            elif close[i] < short_breakout and volume_filter[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when price falls below previous low - ATR
            if close[i] < short_breakout:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when price rises above previous high + ATR
            if close[i] > long_breakout:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals