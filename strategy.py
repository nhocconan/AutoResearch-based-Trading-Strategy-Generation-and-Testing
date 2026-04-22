#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h trend following with 4h Supertrend filter and volume confirmation.
# Long when price > 4h Supertrend + volume > 1.5x 20-period average.
# Short when price < 4h Supertrend + volume > 1.5x 20-period average.
# Exit when price crosses back below/above 4h Supertrend.
# Uses 4h for trend direction, 1h for entry timing and volume confirmation.
# Designed for low trade frequency (15-30/year) to avoid fee drag in 1h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 4h data for Supertrend calculation
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate ATR for Supertrend (period=10)
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Calculate Supertrend
    hl2 = (high_4h + low_4h) / 2
    upper = hl2 + (3.0 * atr)
    lower = hl2 - (3.0 * atr)
    
    supertrend = np.zeros_like(close_4h)
    direction = np.ones_like(close_4h)  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = upper[0]
    direction[0] = 1
    
    for i in range(1, len(close_4h)):
        if close_4h[i] > supertrend[i-1]:
            supertrend[i] = max(upper[i], supertrend[i-1])
            direction[i] = 1
        else:
            supertrend[i] = min(lower[i], supertrend[i-1])
            direction[i] = -1
    
    # Align Supertrend to 1h
    supertrend_aligned = align_htf_to_ltf(prices, df_4h, supertrend)
    
    # Volume confirmation (20-period average)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(supertrend_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        st = supertrend_aligned[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        vol_filter = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long: price above Supertrend + volume confirmation
            if price > st and vol_filter:
                signals[i] = 0.20
                position = 1
            # Short: price below Supertrend + volume confirmation
            elif price < st and vol_filter:
                signals[i] = -0.20
                position = -1
        
        elif position != 0:
            # Exit: price crosses back through Supertrend
            exit_signal = False
            
            if position == 1:  # long position
                if price < st:
                    exit_signal = True
            
            elif position == -1:  # short position
                if price > st:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1h_Supertrend4h_Volume_Filter"
timeframe = "1h"
leverage = 1.0