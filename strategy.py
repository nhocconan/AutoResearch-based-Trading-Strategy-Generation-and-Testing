#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_Volume_ATRFilter_V1
Hypothesis: Camarilla pivot R1/S1 breakout on daily timeframe with volume confirmation (>1.5x 20-day average) and ATR-based stoploss works for BTC and ETH in both bull and bear markets. Uses weekly timeframe only for HTF alignment (not for signal generation). Target: 7-25 trades/year per symbol (30-100 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data once (primary timeframe)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Camarilla pivot points for 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: R1 = close + (high - low) * 1.1/12, S1 = close - (high - low) * 1.1/12
    rng = high_1d - low_1d
    r1 = close_1d + rng * (1.1 / 12)
    s1 = close_1d - rng * (1.1 / 12)
    
    # Align Camarilla levels to 1d timeframe (no shift needed as we use same timeframe)
    r1_aligned = r1  # already on 1d
    s1_aligned = s1  # already on 1d
    
    # Volume filter: 20-day average
    vol_ma = df_1d['volume'].rolling(window=20, min_periods=20).mean().values
    
    # ATR for stoploss (using 1d data)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(30, n):  # start after warmup for indicators
        # Skip if indicators not ready
        if (np.isnan(vol_ma[i]) or np.isnan(atr[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        price = df_1d['close'].iloc[i]
        volume = df_1d['volume'].iloc[i]
        
        # Volume confirmation (>1.5x average)
        volume_ok = volume > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume
            if price > r1_aligned[i] and volume_ok:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below S1 with volume
            elif price < s1_aligned[i] and volume_ok:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Exit: price closes below S1 or ATR stoploss
            if price < s1_aligned[i] or price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price closes above R1 or ATR stoploss
            if price > r1_aligned[i] or price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_Volume_ATRFilter_V1"
timeframe = "1d"
leverage = 1.0