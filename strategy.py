#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with 1d ATR filter and volume confirmation.
- Williams Alligator: Jaw (EMA13, 8-bar shift), Teeth (EMA8, 5-bar shift), Lips (EMA5, 3-bar shift)
- Long: Lips > Teeth > Jaw (bullish alignment) + ATR(14) < ATR(50) (low volatility) + volume > 1.3x 20-period avg
- Short: Lips < Teeth < Jaw (bearish alignment) + ATR(14) < ATR(50) (low volatility) + volume > 1.3x 20-period avg
- Exit: Opposite Alligator alignment OR volatility expansion (ATR(14) > 1.5x ATR(50))
- 1d ATR filter ensures trades occur during low volatility regimes to avoid whipsaws
- Volume confirmation ensures participation in the breakout
- Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag on 12h timeframe
- Works in both bull (trend continuation via Alligator alignment) and bear (mean reversion via volatility contraction/expansion)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume confirmation: > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Williams Alligator components (using 12h data)
    jaw = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().shift(8).values  # Jaw: EMA13, 8-bar shift
    teeth = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().shift(5).values   # Teeth: EMA8, 5-bar shift
    lips = pd.Series(close).ewm(span=5, adjust=False, min_periods=5).mean().shift(3).values    # Lips: EMA5, 3-bar shift
    
    # ATR calculation for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # Load 1d data ONCE before loop for ATR filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d ATR calculation
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr1_1d[0] = 0
    tr2_1d[0] = 0
    tr3_1d[0] = 0
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    atr_14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_50_1d = pd.Series(tr_1d).rolling(window=50, min_periods=50).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    atr_50_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 13, 8, 5, 14, 50)  # Need 20 for volume MA, 13/8/5 for Alligator, 14/50 for ATR
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(jaw[i]) or 
            np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or 
            np.isnan(atr_14[i]) or 
            np.isnan(atr_50[i]) or 
            np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(atr_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike confirmation (> 1.3x average)
        volume_spike = volume[i] > 1.3 * vol_ma[i]
        
        # Williams Alligator alignment
        bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
        bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # Volatility filter: low volatility regime (ATR(14) < ATR(50))
        low_volatility = atr_14[i] < atr_50[i]
        
        # 1d ATR filter: ensure low volatility on higher timeframe
        low_volatility_1d = atr_14_1d_aligned[i] < atr_50_1d_aligned[i]
        
        if position == 0:
            # Long: Bullish Alligator alignment + low volatility (both TFs) + volume spike
            if volume_spike and bullish_alignment and low_volatility and low_volatility_1d:
                signals[i] = 0.25
                position = 1
            # Short: Bearish Alligator alignment + low volatility (both TFs) + volume spike
            elif volume_spike and bearish_alignment and low_volatility and low_volatility_1d:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bearish Alligator alignment OR volatility expansion (ATR(14) > 1.5x ATR(50)) OR 1d volatility expansion
            volatility_expansion = atr_14[i] > 1.5 * atr_50[i]
            volatility_expansion_1d = atr_14_1d_aligned[i] > 1.5 * atr_50_1d_aligned[i]
            if bearish_alignment or volatility_expansion or volatility_expansion_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bullish Alligator alignment OR volatility expansion (ATR(14) > 1.5x ATR(50)) OR 1d volatility expansion
            volatility_expansion = atr_14[i] > 1.5 * atr_50[i]
            volatility_expansion_1d = atr_14_1d_aligned[i] > 1.5 * atr_50_1d_aligned[i]
            if bullish_alignment or volatility_expansion or volatility_expansion_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_1dATR_VolumeSpike"
timeframe = "12h"
leverage = 1.0