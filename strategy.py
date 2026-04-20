#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for pivot points and ATR
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily Pivot Point, R1, S1
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = 2 * pivot_1d - low_1d
    s1_1d = 2 * pivot_1d - high_1d
    
    # Align to 4h (previous day's levels available at 4h open)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # 1d ATR for volatility filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 4h price and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume confirmation: 4-period average (1 day of 4h bars)
    vol_ma = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if NaN in critical values
        if np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or np.isnan(atr_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long: break above R1 with volume and sufficient volatility
            if price > r1_1d_aligned[i] and vol > 1.5 * vol_ma[i] and atr_1d_aligned[i] > 0:
                signals[i] = 0.20
                position = 1
                entry_price = price
            # Short: break below S1 with volume and sufficient volatility
            elif price < s1_1d_aligned[i] and vol > 1.5 * vol_ma[i] and atr_1d_aligned[i] > 0:
                signals[i] = -0.20
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: price crosses below S1 or volatility drops
            if price < s1_1d_aligned[i] or vol < 0.8 * vol_ma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: price crosses above R1 or volatility drops
            if price > r1_1d_aligned[i] or vol < 0.8 * vol_ma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "4h_PivotPoint_R1S1_Breakout_Volume_ATRFilter_Tight"
timeframe = "4h"
leverage = 1.0