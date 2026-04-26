#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_ATRVolRegime
Hypothesis: 4h Donchian(20) breakout with ATR-based volatility regime filter and volume confirmation.
Only long when price breaks above upper band in low volatility regime (ATR ratio < 0.8) with volume spike.
Only short when price breaks below lower band in low volatility regime with volume spike.
Uses discrete position sizing (0.25) to minimize fee drag. Target: 20-50 trades/year per symbol.
Volatility regime filter prevents whipsaws in high volatility choppy markets.
Works in bull/bear via symmetric breakout logic.
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
    
    # Calculate ATR(14) for volatility regime
    tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    tr = np.concatenate([[np.nan], tr])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate ATR ratio (current ATR / 50-period MA of ATR) for volatility regime
    atr_ma = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr / atr_ma
    low_vol_regime = atr_ratio < 0.8  # Low volatility regime
    
    # Volume spike detector (20-bar volume MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    # Donchian channels (20-period)
    upper_channel = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lower_channel = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(low_vol_regime[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:
            # Long: price breaks above upper channel in low vol regime with volume spike
            if close[i] > upper_channel[i] and low_vol_regime[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower channel in low vol regime with volume spike
            elif close[i] < lower_channel[i] and low_vol_regime[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price closes below upper channel OR volatility increases OR volume drops
            if close[i] < upper_channel[i] or not low_vol_regime[i] or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price closes above lower channel OR volatility increases OR volume drops
            if close[i] > lower_channel[i] or not low_vol_regime[i] or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_ATRVolRegime"
timeframe = "4h"
leverage = 1.0