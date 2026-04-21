#!/usr/bin/env python3
"""
6h_Camarilla_R1S1_Breakout_Volume_EMA34Filter_v2
Hypothesis: 6h Camarilla R1/S1 breakout with volume confirmation (>1.5x 20-bar MA) and 1d EMA34 trend filter works in both bull and bear markets. Uses 1d timeframe for EMA34 to avoid look-ahead. Target: 12-37 trades/year per symbol (50-150 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data once for EMA34 trend (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Camarilla pivot levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = df_1d['close'].values
    
    # Shift by 1 to use previous day's data for today's pivot
    high_1d_prev = np.roll(high_1d, 1)
    low_1d_prev = np.roll(low_1d, 1)
    close_1d_prev = np.roll(close_1d, 1)
    high_1d_prev[0] = low_1d_prev[0] = close_1d_prev[0] = np.nan
    
    # Camarilla calculations
    rangeprev = high_1d_prev - low_1d_prev
    R1 = close_1d_prev + rangeprev * 1.1 / 12
    S1 = close_1d_prev - rangeprev * 1.1 / 12
    R4 = close_1d_prev + rangeprev * 1.1 / 2
    S4 = close_1d_prev - rangeprev * 1.1 / 2
    
    # Align pivot levels to 6h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    
    # Volume filter: 20-period average
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    # ATR for stoploss
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or np.isnan(R4_aligned[i]) or 
            np.isnan(S4_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation (>1.5x average)
        volume_ok = volume > 1.5 * vol_ma[i]
        
        # 1d EMA34 trend filter
        uptrend = price > ema34_1d_aligned[i]
        downtrend = price < ema34_1d_aligned[i]
        
        if position == 0:
            # Long: break above R1 with volume and uptrend
            if price > R1_aligned[i] and volume_ok and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with volume and downtrend
            elif price < S1_aligned[i] and volume_ok and downtrend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price crosses below S1 or stoploss
            if price < S1_aligned[i] or price < prices['close'].iloc[i-1] - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price crosses above R1 or stoploss
            if price > R1_aligned[i] or price > prices['close'].iloc[i-1] + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R1S1_Breakout_Volume_EMA34Filter_v2"
timeframe = "6h"
leverage = 1.0