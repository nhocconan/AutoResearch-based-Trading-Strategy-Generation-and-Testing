#!/usr/bin/env python3
# 6h_1d_Keltner_Channel_Breakout_Volume
# Hypothesis: On 6h timeframe, trade breakouts from daily Keltner Channel (EMA20 + ATR10*2) with volume confirmation.
# Uses daily volatility expansion to filter breakouts. Works in both bull and bear markets due to volatility-based entries.
# Target: 20-50 trades per year by requiring volatility expansion + volume spike.

name = "6h_1d_Keltner_Channel_Breakout_Volume"
timeframe = "6h"
leverage = 1.0

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
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily Keltner Channel: EMA20 ± ATR(10)*2
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # EMA20
    ema20 = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # ATR(10)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr10 = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Keltner Bands
    upper_keltner = ema20 + 2 * atr10
    lower_keltner = ema20 - 2 * atr10
    
    # Align daily Keltner levels to 6h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper_keltner)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower_keltner)
    
    # Volume average for spike detection (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long breakout above upper Keltner with volume confirmation
            if (close[i] > upper_aligned[i] * 1.002 and 
                volume[i] > 1.8 * volume_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short breakdown below lower Keltner with volume
            elif (close[i] < lower_aligned[i] * 0.998 and 
                  volume[i] > 1.8 * volume_ma[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: breakdown below lower Keltner
            if close[i] < lower_aligned[i] * 0.998:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: breakout above upper Keltner
            if close[i] > upper_aligned[i] * 1.002:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals