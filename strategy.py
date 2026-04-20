#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily ATR(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Calculate daily Donchian(20) channels
    donch_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    
    # Calculate daily volume average for confirmation
    volume_1d = df_1d['volume'].values
    vol_avg_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get values
        close_val = prices['close'].iloc[i]
        atr_val = atr_14_aligned[i]
        upper_val = donch_high_aligned[i]
        lower_val = donch_low_aligned[i]
        vol_val = volume_1d[i]
        vol_avg_val = vol_avg_20_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(atr_val) or np.isnan(upper_val) or 
            np.isnan(lower_val) or np.isnan(vol_avg_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above upper Donchian + volume confirmation
            if close_val > upper_val and vol_val > vol_avg_val:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian + volume confirmation
            elif close_val < lower_val and vol_val > vol_avg_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price closes below lower Donchian or ATR-based stop
            if close_val < lower_val or close_val < prices['close'].iloc[i-1] - 1.5 * atr_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price closes above upper Donchian or ATR-based stop
            if close_val > upper_val or close_val > prices['close'].iloc[i-1] + 1.5 * atr_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# 4h_Donchian20_Volume_ATRStop_V1
# Uses daily Donchian(20) channels for breakout signals
# Enters long when price breaks above upper channel with volume above average
# Enters short when price breaks below lower channel with volume above average
# Exits on Donchian reversal or ATR-based stop (1.5x ATR)
# Designed for 4h timeframe with ~20-40 trades/year
name = "4h_Donchian20_Volume_ATRStop_V1"
timeframe = "4h"
leverage = 1.0