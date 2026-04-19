#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with 1d volatility-adjusted breakout and volume confirmation.
# Uses 1d ATR scaled to define breakout levels from the previous 12h candle's high/low.
# Breakout occurs when price moves beyond (high/low ± k * ATR) with volume > 1.5x average.
# Designed to capture momentum bursts in both bull and bear markets with controlled risk.
# Target: 50-150 total trades over 4 years (12-37/year).
name = "12h_1d_VolatilityBreakout_VolumeFilter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR calculation (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range and ATR on 1d timeframe
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period TR is just high-low
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align ATR to 12h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Volume filter: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for ATR
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(atr_1d_aligned[i]) or np.isnan(volume_ma[i]):
            signals[i] = 0.0
            continue
            
        # Calculate breakout levels using previous 12h period high/low + ATR
        prev_high = high[i-1]
        prev_low = low[i-1]
        atr_val = atr_1d_aligned[i]
        
        # Volatility multiplier: 0.5 * ATR for breakout threshold
        long_breakout = prev_high + 0.5 * atr_val
        short_breakout = prev_low - 0.5 * atr_val
        
        if position == 0:
            # Long when price breaks above previous high + 0.5*ATR with volume
            if close[i] > long_breakout and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below previous low - 0.5*ATR with volume
            elif close[i] < short_breakout and volume_filter[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when price falls below previous low - 0.5*ATR
            if close[i] < short_breakout:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when price rises above previous high + 0.5*ATR
            if close[i] > long_breakout:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals