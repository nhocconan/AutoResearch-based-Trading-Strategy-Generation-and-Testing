#!/usr/bin/env python3
"""
4h_HTF_1d_RSI_MeanReversion_With_Volume_Filter
Mean reversion using daily RSI extremes + 4h volume confirmation
Long: daily RSI < 30 AND 4h volume > 1.5x 20-period average
Short: daily RSI > 70 AND 4h volume > 1.5x 20-period average
Exit: opposite RSI extreme or volume drops below threshold
Designed to work in both bull and bear markets by fading extremes
Target: 50-150 total trades over 4 years (12-37/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for RSI calculation
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 14-period RSI on daily timeframe
    delta = pd.Series(close_1d).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
    rs = gain / loss
    rsi_14 = 100 - (100 / (1 + rs))
    rsi_14_values = rsi_14.fillna(50).values  # fill NaN with 50 (neutral)
    
    # Align daily RSI to 4h timeframe
    rsi_14_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_values)
    
    # Calculate 4h volume SMA(20) for volume filter
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = max(30, 20)  # need RSI and volume SMA
    
    for i in range(start_idx, n):
        if (np.isnan(rsi_14_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        vol = volume[i]
        vol_sma_val = vol_sma_20[i]
        rsi = rsi_14_aligned[i]
        
        if position == 0:
            # Long: daily RSI oversold (<30) + volume confirmation
            if rsi < 30 and vol > 1.5 * vol_sma_val:
                signals[i] = 0.25
                position = 1
            # Short: daily RSI overbought (>70) + volume confirmation
            elif rsi > 70 and vol > 1.5 * vol_sma_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI returns to neutral (>50) or volume drops
            if rsi > 50 or vol < vol_sma_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI returns to neutral (<50) or volume drops
            if rsi < 50 or vol < vol_sma_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_HTF_1d_RSI_MeanReversion_With_Volume_Filter"
timeframe = "4h"
leverage = 1.0