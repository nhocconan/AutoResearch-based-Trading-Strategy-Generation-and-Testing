#!/usr/bin/env python3
"""
12h_supertrend_1d_trend_volume_v1
Hypothesis: Supertrend on 12h with direction from 1d EMA200 and volume confirmation.
Long when Supertrend flips up, price above 1d EMA200, and volume > 1.5x average.
Short when Supertrend flips down, price below 1d EMA200, and volume > 1.5x average.
Supertrend avoids whipsaws, EMA200 filters trend, volume confirms strength.
Target: 15-30 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_supertrend_1d_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA200 for trend filter
    ema_200 = df_1d['close'].ewm(span=200, adjust=False).mean()
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200.values)
    
    # Supertrend on 12h (ATR=10, multiplier=3)
    atr_period = 10
    multiplier = 3
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0  # First period has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR
    atr = np.zeros(n)
    atr[atr_period] = np.mean(tr[:atr_period])
    for i in range(atr_period + 1, n):
        atr[i] = (atr[i-1] * (atr_period - 1) + tr[i]) / atr_period
    
    # Supertrend calculation
    hl2 = (high + low) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    supertrend = np.zeros(n)
    supertrend_direction = np.ones(n)  # 1 for uptrend, -1 for downtrend
    
    # Initialize
    supertrend[0] = upper_band[0]
    supertrend_direction[0] = 1
    
    for i in range(1, n):
        if close[i] > supertrend[i-1]:
            supertrend_direction[i] = 1
        elif close[i] < supertrend[i-1]:
            supertrend_direction[i] = -1
        else:
            supertrend_direction[i] = supertrend_direction[i-1]
        
        if supertrend_direction[i] == 1:
            supertrend[i] = max(lower_band[i], supertrend[i-1])
        else:
            supertrend[i] = min(upper_band[i], supertrend[i-1])
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema_200_aligned[i]) or np.isnan(vol_ma[i]) or 
            vol_ma[i] <= 0 or np.isnan(supertrend[i]) or np.isnan(supertrend_direction[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: Supertrend flips down OR price breaks below EMA200
            if supertrend_direction[i] == -1 or close[i] < ema_200_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: Supertrend flips up OR price breaks above EMA200
            if supertrend_direction[i] == 1 or close[i] > ema_200_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Supertrend uptrend with volume and price above EMA200
            if (supertrend_direction[i] == 1 and vol_confirm and 
                close[i] > ema_200_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Supertrend downtrend with volume and price below EMA200
            elif (supertrend_direction[i] == -1 and vol_confirm and 
                  close[i] < ema_200_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals