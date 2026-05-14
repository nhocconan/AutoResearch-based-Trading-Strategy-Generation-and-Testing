#!/usr/bin/env python3
"""
4h_HTF_1d_Camarilla_R1S1_Breakout_ATR_Trail_V1
Hypothesis: Use 1d Camarilla R1/S1 breakouts with volume confirmation and ATR-based trailing stop (not fixed stop) to capture trends while limiting drawdowns in both bull and bear markets. 
Trailing stop adapts to volatility, reducing whipsaw in ranging markets and locking in profits during strong trends. 
Position size fixed at 0.30 for consistency. Target 20-40 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')  # for 1d Camarilla levels
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # === 1d Camarilla Pivot Levels (R1, S1) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point
    pivot = (high_1d + low_1d + close_1d) / 3.0
    # Camarilla levels
    r1 = close_1d + (high_1d - low_1d) * 1.1 / 12.0
    s1 = close_1d - (high_1d - low_1d) * 1.1 / 12.0
    
    # Align to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === 4h Indicators ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume MA (20-period) for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high = 0.0  # for long trailing stop
    lowest_low = 0.0    # for short trailing stop
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) 
            or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ok = vol > 2.0 * vol_ma[i]  # volume spike confirmation
        
        if position == 0:
            # Long: break above 1d Camarilla R1 with volume spike
            if price > r1_aligned[i-1] and vol_ok:
                signals[i] = 0.30
                position = 1
                highest_high = price
            # Short: break below 1d Camarilla S1 with volume spike
            elif price < s1_aligned[i-1] and vol_ok:
                signals[i] = -0.30
                position = -1
                lowest_low = price
        
        elif position == 1:
            # Update highest high for trailing stop
            if price > highest_high:
                highest_high = price
            # ATR trailing stop: exit if price drops 2.5*ATR from highest high
            if price < highest_high - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:
            # Update lowest low for trailing stop
            if price < lowest_low:
                lowest_low = price
            # ATR trailing stop: exit if price rises 2.5*ATR from lowest low
            if price > lowest_low + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_HTF_1d_Camarilla_R1S1_Breakout_ATR_Trail_V1"
timeframe = "4h"
leverage = 1.0