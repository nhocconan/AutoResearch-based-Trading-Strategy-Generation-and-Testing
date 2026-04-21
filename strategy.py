#!/usr/bin/env python3
"""
4h Donchian(20) Breakout with Volume Confirmation and ATR Stop
Hypothesis: Donchian channel breakouts capture strong directional moves.
Volume confirmation filters false breakouts. ATR-based stops limit losses.
Works in bull/bear markets by using volatility-based channels and avoiding overtrading.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for ATR and volume baseline
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily True Range for ATR
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Main timeframe data (4h)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period)
    high_max = np.full(n, np.nan)
    low_min = np.full(n, np.nan)
    for i in range(20, n):
        high_max[i] = np.max(high[i-20:i])
        low_min[i] = np.min(low[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if NaN in critical values
        if np.isnan(high_max[i]) or np.isnan(low_min[i]) or np.isnan(atr_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        upper = high_max[i]
        lower = low_min[i]
        atr = atr_1d_aligned[i]
        vol_current = volume[i]
        
        # Volume filter: current volume > 1.5x 20-period average
        vol_ma = np.mean(volume[max(0, i-20):i]) if i >= 20 else volume[i]
        vol_ok = vol_current > 1.5 * vol_ma
        
        if position == 0:
            # Long breakout: price breaks above upper band with volume confirmation
            if price > upper and vol_ok:
                signals[i] = 0.30
                position = 1
            # Short breakdown: price breaks below lower band with volume confirmation
            elif price < lower and vol_ok:
                signals[i] = -0.30
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below lower band or ATR-based stop
            if price < lower or (i > 0 and close[i-1] > lower and price < close[i-1] - 1.5 * atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:
            # Short exit: price breaks above upper band or ATR-based stop
            if price > upper or (i > 0 and close[i-1] < upper and price > close[i-1] + 1.5 * atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_Donchian20_Volume_ATRFilter"
timeframe = "4h"
leverage = 1.0