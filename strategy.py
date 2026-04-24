#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla H3/L3 breakout with 1d ATR regime filter and volume confirmation.
- Uses 1d ATR to detect trending (ATR > 20-period SMA) vs ranging markets.
- Only takes breakouts in trending regimes to avoid false breakouts in chop.
- Volume spike (>1.5x 20-period average) confirms breakout validity.
- Position size 0.25 to limit drawdown and reduce fee churn.
- Designed to work in both bull and bear markets via regime filter.
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
    
    # Get 1d data ONCE before loop for Camarilla levels, ATR, and regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels from previous completed daily bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_h3 = close_1d + (high_1d - low_1d) * 1.1 / 4
    camarilla_l3 = close_1d - (high_1d - low_1d) * 1.1 / 4
    
    # 1d ATR (14-period) and its 20-period SMA for regime detection
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr1 = np.maximum(tr1, np.abs(low_1d[1:] - close_1d[:-1]))
    tr1 = np.concatenate([[np.nan], tr1])  # align length
    atr_14 = pd.Series(tr1).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_ma = pd.Series(atr_14).rolling(window=20, min_periods=20).mean().values
    trending_regime = atr_14 > atr_ma  # True when ATR > its MA (trending market)
    
    # Align all 1d indicators to 12h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    trending_regime_aligned = align_htf_to_ltf(prices, df_1d, trending_regime.astype(float))
    
    # Volume confirmation: > 1.5x 20-period average volume (less strict than 2.0x)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(trending_regime_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above Camarilla H3 with volume spike and in trending regime
            if close[i] > camarilla_h3_aligned[i] and volume_spike[i] and trending_regime_aligned[i] > 0.5:
                signals[i] = 0.25
                position = 1
            # Short: break below Camarilla L3 with volume spike and in trending regime
            elif close[i] < camarilla_l3_aligned[i] and volume_spike[i] and trending_regime_aligned[i] > 0.5:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below Camarilla L3 OR regime turns ranging
            if close[i] < camarilla_l3_aligned[i] or trending_regime_aligned[i] <= 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above Camarilla H3 OR regime turns ranging
            if close[i] > camarilla_h3_aligned[i] or trending_regime_aligned[i] <= 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_1dATR_Regime_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0