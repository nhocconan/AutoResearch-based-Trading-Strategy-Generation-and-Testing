#!/usr/bin/env python3
"""
Hypothesis: 6h Williams Alligator + 1d ATR Regime + Volume Confirmation
- Long when: Alligator bullish (jaw < teeth < lips), ATR regime = trending (ATR(7)/ATR(30) > 1.2), volume > 1.5x 20-period average
- Short when: Alligator bearish (jaw > teeth > lips), ATR regime = trending (ATR(7)/ATR(30) > 1.2), volume > 1.5x 20-period average
- Exit when Alligator reverses (jaws cross teeth) or ATR regime becomes choppy (ATR(7)/ATR(30) < 0.8)
- Uses 1d HTF for ATR regime filter to avoid whipsaw in ranging markets
- Williams Alligator: jaw=SMA(13,8), teeth=SMA(8,5), lips=SMA(5,3) - smoothed with SMMA (using EMA as proxy)
- Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag
- Designed to work in both bull and bear markets via ATR regime filter
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams Alligator (using EMA as proxy for SMMA)
    jaw = pd.Series(close).ewm(span=13, adjust=False).mean().values  # SMA(13,8) -> EMA(13)
    teeth = pd.Series(close).ewm(span=8, adjust=False).mean().values   # SMA(8,5) -> EMA(8)
    lips = pd.Series(close).ewm(span=5, adjust=False).mean().values    # SMA(5,3) -> EMA(5)
    
    # Get 1d data ONCE before loop for ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ATR(7) and ATR(30) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.max([high_1d[0] - low_1d[0], np.abs(high_1d[0] - close_1d[0]), np.abs(low_1d[0] - close_1d[0])])], 
                            np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr_7_1d = pd.Series(tr_1d).ewm(span=7, adjust=False, min_periods=7).mean().values
    atr_30_1d = pd.Series(tr_1d).ewm(span=30, adjust=False, min_periods=30).mean().values
    
    # ATR ratio regime: >1.2 = trending, <0.8 = choppy
    atr_ratio_1d = atr_7_1d / atr_30_1d
    atr_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio_1d)
    
    # Volume confirmation: > 1.5x 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(13, 8, 5, 30, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(atr_ratio_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Alligator bullish + trending regime + volume spike
            if jaw[i] < teeth[i] and teeth[i] < lips[i] and atr_ratio_1d_aligned[i] > 1.2 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Alligator bearish + trending regime + volume spike
            elif jaw[i] > teeth[i] and teeth[i] > lips[i] and atr_ratio_1d_aligned[i] > 1.2 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Alligator reverses OR regime becomes choppy
            if not (jaw[i] < teeth[i] and teeth[i] < lips[i]) or atr_ratio_1d_aligned[i] < 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator reverses OR regime becomes choppy
            if not (jaw[i] > teeth[i] and teeth[i] > lips[i]) or atr_ratio_1d_aligned[i] < 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsAlligator_1dATRRegime_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0