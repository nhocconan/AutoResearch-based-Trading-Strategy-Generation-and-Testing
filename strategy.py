#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d ATR regime filter and volume confirmation.
Uses Donchian channel breakouts for trend capture, combined with 1d ATR-based regime filter
to avoid whipsaws in choppy markets. Volume spike confirms breakout strength.
Designed for 12h timeframe to capture swing moves with minimal trade frequency.
Target: 12-37 trades/year per symbol (50-150 total over 4 years).
Uses discrete position sizing (0.25) to minimize fee churn.
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
    tr1[0] = high_1d[0] - low_1d[0]  # first bar has no previous close
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate ATR regime: ATR > 20-period mean = high volatility (trending)
    atr_ma_20 = pd.Series(atr_14_1d_aligned).rolling(window=20, min_periods=20).mean().values
    high_vol_regime = atr_14_1d_aligned > atr_ma_20
    
    # Calculate Donchian(20) channels on 12h data
    donchian_hi = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lo = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume spike: current volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 20)  # need Donchian20 and volume MA20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_hi[i]) or np.isnan(donchian_lo[i]) or 
            np.isnan(atr_14_1d_aligned[i]) or np.isnan(atr_ma_20[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above Donchian high AND high volatility regime AND volume spike
            if close[i] > donchian_hi[i] and high_vol_regime[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian low AND high volatility regime AND volume spike
            elif close[i] < donchian_lo[i] and high_vol_regime[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: break of opposite Donchian level
            exit_signal = False
            if position == 1:
                # Exit long on break below Donchian low
                if close[i] < donchian_lo[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short on break above Donchian high
                if close[i] > donchian_hi[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Donchian20_Breakout_1dATRRegime_VolumeConfirmation"
timeframe = "12h"
leverage = 1.0