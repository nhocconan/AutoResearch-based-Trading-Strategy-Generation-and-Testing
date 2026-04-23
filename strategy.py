#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian channel breakout with 1d ATR regime filter and volume spike confirmation.
In low volatility regimes (ATR contraction), breakouts from Donchian channels often fail.
In high volatility regimes (ATR expansion), breakouts tend to sustain. Using 1d ATR to classify
regime avoids using current-bar volatility which can be misleading. Volume spike confirms
institutional participation. Works in bull/bear via volatility regime alignment.
Target: 12-37 trades/year per symbol (50-150 total over 4 years).
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
    
    # Calculate 20-period Donchian channels
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d ATR(14) for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Set first TR to high-low (no previous close)
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Calculate 1d ATR(50) for regime classification (longer term volatility)
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    atr_50_aligned = align_htf_to_ltf(prices, df_1d, atr_50)
    
    # Volatility regime: ATR(14) > ATR(50) = expanding volatility (good for breakouts)
    vol_regime_expanding = atr_14_aligned > atr_50_aligned
    
    # Calculate volume spike: 6h volume > 2.0 x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50, 20)  # Donchian20, ATR50, volMA20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(atr_14_aligned[i]) or np.isnan(atr_50_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above Donchian high AND expanding volatility AND volume spike
            if close[i] > high_20[i] and vol_regime_expanding[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian low AND expanding volatility AND volume spike
            elif close[i] < low_20[i] and vol_regime_expanding[i] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: break of opposite Donchian level
            exit_signal = False
            if position == 1:
                # Exit long on break below Donchian low
                if close[i] < low_20[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short on break above Donchian high
                if close[i] > high_20[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Donchian20_Breakout_1dATR_Regime_VolumeSpike"
timeframe = "6h"
leverage = 1.0