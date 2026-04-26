#!/usr/bin/env python3
"""
4h_TRIX_VolumeSpike_ChopRegime
Hypothesis: On 4h timeframe, enter long when TRIX crosses above zero AND volume > 1.8x 20-period average volume AND chop > 61.8 (range regime). Enter short when TRIX crosses below zero AND volume > 1.8x 20-period average volume AND chop > 61.8. Exit on opposite TRIX cross. Uses discrete sizing (0.0, ±0.25) to limit fee drag. Target: 20-50 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate TRIX (15-period EMA of EMA of EMA of close, then ROC)
    # TRIX = 100 * (EMA3(close) - EMA3(close)_prev) / EMA3(close)_prev
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix = 100 * (ema3 - np.roll(ema3, 1)) / np.roll(ema3, 1)
    trix[0] = 0  # first value undefined
    
    # Calculate Choppiness Index (14-period)
    # CHOP = 100 * log10(sum(ATR(1)) / (nperiods * log(nperiods))) / log10(nperiods)
    tr1 = np.maximum(high - low, np.roll(np.abs(high - np.roll(close, 1)), 1))
    tr1 = np.maximum(tr1, np.roll(np.abs(low - np.roll(close, 1)), 1))
    tr1[0] = high[0] - low[0]  # first TR
    atr_sum = pd.Series(tr1).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(atr_sum / (14 * np.log10(14))) / np.log10(14)
    
    # Volume confirmation: fixed threshold of 1.8x average volume
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.8 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need TRIX, CHOP, and volume MA warmup
    start_idx = max(15*3 + 1, 14, 20)  # TRIX needs ~45, CHOP needs 14, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(trix[i]) or np.isnan(chop[i]) or np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # TRIX cross signals
        trix_cross_up = trix[i] > 0 and trix[i-1] <= 0
        trix_cross_down = trix[i] < 0 and trix[i-1] >= 0
        
        # Regime filter: chop > 61.8 = ranging market (good for mean reversion)
        in_chop_regime = chop[i] > 61.8
        
        if position == 0:
            # Long: TRIX cross above zero + volume spike + chop regime
            long_signal = trix_cross_up and volume_spike[i] and in_chop_regime
            
            # Short: TRIX cross below zero + volume spike + chop regime
            short_signal = trix_cross_down and volume_spike[i] and in_chop_regime
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: TRIX cross below zero
            if trix_cross_down:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: TRIX cross above zero
            if trix_cross_up:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_TRIX_VolumeSpike_ChopRegime"
timeframe = "4h"
leverage = 1.0