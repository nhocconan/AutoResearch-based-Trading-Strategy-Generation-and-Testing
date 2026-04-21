#!/usr/bin/env python3
"""
1d_HTF_1w_Camarilla_R1S1_Breakout_ATR_Trail_V1
Hypothesis: Use 1w Camarilla R1/S1 breakouts on the daily chart with volume confirmation and ATR-based trailing stop to capture major trends while minimizing whipsaw. The weekly timeframe provides stronger structural levels that are more reliable in both bull and bear markets, reducing false breakouts. Position size fixed at 0.25 to balance return and drawdown. Target 15-25 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')  # for 1w Camarilla levels
    
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # === 1w Camarilla Pivot Levels (R1, S1) ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Pivot point
    pivot = (high_1w + low_1w + close_1w) / 3.0
    # Camarilla levels
    r1 = close_1w + (high_1w - low_1w) * 1.1 / 12.0
    s1 = close_1w - (high_1w - low_1w) * 1.1 / 12.0
    
    # Align to 1d timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # === 1d Indicators ===
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
        vol_ok = vol > 1.5 * vol_ma[i]  # volume confirmation
        
        if position == 0:
            # Long: break above 1w Camarilla R1 with volume confirmation
            if price > r1_aligned[i-1] and vol_ok:
                signals[i] = 0.25
                position = 1
                highest_high = price
            # Short: break below 1w Camarilla S1 with volume confirmation
            elif price < s1_aligned[i-1] and vol_ok:
                signals[i] = -0.25
                position = -1
                lowest_low = price
        
        elif position == 1:
            # Update highest high for trailing stop
            if price > highest_high:
                highest_high = price
            # ATR trailing stop: exit if price drops 2.0*ATR from highest high
            if price < highest_high - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Update lowest low for trailing stop
            if price < lowest_low:
                lowest_low = price
            # ATR trailing stop: exit if price rises 2.0*ATR from lowest low
            if price > lowest_low + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_HTF_1w_Camarilla_R1S1_Breakout_ATR_Trail_V1"
timeframe = "1d"
leverage = 1.0