#!/usr/bin/env python3
"""
6H Supertrend with Volume Confirmation and 1D Trend Filter
Long when price closes above Supertrend with expanding volume AND 1D EMA trend up
Short when price closes below Supertrend with expanding volume AND 1D EMA trend down
Exit when price closes back inside Supertrend band
Uses ATR-based trend following that adapts to volatility, reducing whipsaws in ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_supertrend_volume_1d_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === ATR (10) for Supertrend ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # === Supertrend (10, 3.0) ===
    hl2 = (high + low) / 2
    upperband = hl2 + 3.0 * atr
    lowerband = hl2 - 3.0 * atr
    
    # Initialize Supertrend arrays
    supertrend = np.full(n, np.nan)
    uptrend = np.full(n, True)
    
    # Calculate Supertrend
    for i in range(1, n):
        if np.isnan(upperband[i-1]) or np.isnan(lowerband[i-1]):
            supertrend[i] = np.nan
            uptrend[i] = True
            continue
            
        if close[i] <= upperband[i-1]:
            upperband[i] = min(upperband[i], upperband[i-1])
        else:
            upperband[i] = upperband[i]
            
        if close[i] >= lowerband[i-1]:
            lowerband[i] = max(lowerband[i], lowerband[i-1])
        else:
            lowerband[i] = lowerband[i]
        
        if supertrend[i-1] == upperband[i-1]:
            if close[i] <= upperband[i]:
                supertrend[i] = upperband[i]
            else:
                supertrend[i] = lowerband[i]
                uptrend[i] = False
        else:
            if close[i] >= lowerband[i]:
                supertrend[i] = lowerband[i]
                uptrend[i] = True
            else:
                supertrend[i] = upperband[i]
                uptrend[i] = False
    
    # === Volume confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)  # Avoid division by zero
    
    # === 1D trend filter (EMA 21) ===
    df_1d = get_htf_data(prices, '1d')
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(10, n):
        if (np.isnan(supertrend[i]) or np.isnan(vol_ratio[i]) or np.isnan(ema_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes back below Supertrend (trend reversal)
            if close[i] < supertrend[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes back above Supertrend (trend reversal)
            if close[i] > supertrend[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need expanding volume (above average)
            if vol_ratio[i] < 1.5:
                signals[i] = 0.0
                continue
            
            # Entry: Supertrend direction with volume confirmation AND 1D trend filter
            if close[i] > supertrend[i] and ema_1d_aligned[i] > ema_1d_aligned[i-1]:
                # Close above Supertrend with rising 1D EMA -> long
                position = 1
                signals[i] = 0.25
            elif close[i] < supertrend[i] and ema_1d_aligned[i] < ema_1d_aligned[i-1]:
                # Close below Supertrend with falling 1D EMA -> short
                position = -1
                signals[i] = -0.25
    
    return signals