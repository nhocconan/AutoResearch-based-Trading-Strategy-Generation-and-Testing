#!/usr/bin/env python3
"""
Hypothesis: 4h Williams Alligator with 1d ATR Regime Filter and Volume Spike.
- Primary timeframe: 4h for execution, HTF: 1d for ATR-based regime filter.
- Entry: Williams Alligator signals (jaw-teeth-lips alignment) with volume > 2.0x 20-period volume MA.
- Regime filter: only trade when 1d ATR(14) > 1.5 * ATR(50) indicating high volatility/trending market.
- Exit: Opposite Alligator signal or ATR regime shift to low volatility.
- Williams Alligator: Jaw (13,8), Teeth (8,5), Lips (5,3) SMAs of median price.
- Discrete signal size: 0.25 to balance return and drawdown control.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
- Works in bull via buying Alligator buy signals in high vol, in bear via selling sell signals in high vol.
- Avoids low-volatility choppy markets where Alligator whipsaws.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Williams Alligator on median price
    median_price = (high + low) / 2.0
    
    # Jaw: 13-period SMMA, shifted 8 bars
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean()
    jaw = jaw.shift(8)
    
    # Teeth: 8-period SMMA, shifted 5 bars
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean()
    teeth = teeth.shift(5)
    
    # Lips: 5-period SMMA, shifted 3 bars
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean()
    lips = lips.shift(3)
    
    jaw_vals = jaw.values
    teeth_vals = teeth.values
    lips_vals = lips.values
    
    # Calculate 1d ATR-based regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value NaN
    
    # ATR(14) and ATR(50)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # Regime: high volatility when ATR(14) > 1.5 * ATR(50)
    atr_regime = atr_14 > (1.5 * atr_50)
    atr_regime_aligned = align_htf_to_ltf(prices, df_1d, atr_regime)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(13, 8, 5, 20) + 8  # Max shift + buffer
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_vals[i]) or np.isnan(teeth_vals[i]) or np.isnan(lips_vals[i]) or
            np.isnan(atr_regime_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Alligator buy: Lips > Teeth > Jaw (aligned, not intertwined)
            if (lips_vals[i] > teeth_vals[i] and teeth_vals[i] > jaw_vals[i] and 
                volume_spike[i] and atr_regime_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Alligator sell: Jaw > Teeth > Lips (aligned, not intertwined)
            elif (jaw_vals[i] > teeth_vals[i] and teeth_vals[i] > lips_vals[i] and 
                  volume_spike[i] and atr_regime_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Alligator sell signal OR regime shift to low volatility
            if (jaw_vals[i] > teeth_vals[i] and teeth_vals[i] > lips_vals[i]) or \
               (~atr_regime_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator buy signal OR regime shift to low volatility
            if (lips_vals[i] > teeth_vals[i] and teeth_vals[i] > jaw_vals[i]) or \
               (~atr_regime_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Williams_Alligator_1dATRRegime_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0