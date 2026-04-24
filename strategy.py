#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla H3/L3 breakout with 1d ATR-based volatility regime filter and volume confirmation.
- Uses 12h timeframe targeting 50-150 total trades over 4 years (12-37/year) to avoid fee drag.
- Volatility regime: ATR(14) > ATR(50) indicates high volatility (trending) environment for breakouts.
- Volume confirmation (>2.0x 20-period average) ensures conviction.
- Works in bull/bear via volatility regime filter - only takes breakouts when volatility is expanding.
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
    
    # Get 1d data ONCE before loop for Camarilla levels, ATR regime, and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels from previous completed daily bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla H3, L3 levels: H3 = C + (H-L)*1.1/4, L3 = C - (H-L)*1.1/4
    camarilla_h3 = close_1d + (high_1d - low_1d) * 1.1 / 4
    camarilla_l3 = close_1d - (high_1d - low_1d) * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe (using previous completed 1d bar)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # 1d ATR-based volatility regime filter: ATR(14) > ATR(50) = expanding volatility (good for breakouts)
    tr_1d = np.maximum(
        high_1d[1:] - low_1d[1:],
        np.maximum(
            np.abs(high_1d[1:] - close_1d[:-1]),
            np.abs(low_1d[1:] - close_1d[:-1])
        )
    )
    tr_1d = np.concatenate([[np.nan], tr_1d])  # align length
    
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_50_1d = pd.Series(tr_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    volatility_regime = atr_14_1d > atr_50_1d  # True when volatility is expanding
    
    # Align volatility regime to 12h timeframe
    volatility_regime_aligned = align_htf_to_ltf(prices, df_1d, volatility_regime.astype(float))
    
    # Volume confirmation: > 2.0x 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(volatility_regime_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above Camarilla H3 with volume spike AND expanding volatility regime
            if close[i] > camarilla_h3_aligned[i] and volume_spike[i] and volatility_regime_aligned[i] > 0.5:
                signals[i] = 0.25
                position = 1
            # Short: break below Camarilla L3 with volume spike AND expanding volatility regime
            elif close[i] < camarilla_l3_aligned[i] and volume_spike[i] and volatility_regime_aligned[i] > 0.5:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below Camarilla L3 OR volatility regime contracts
            if close[i] < camarilla_l3_aligned[i] or volatility_regime_aligned[i] <= 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above Camarilla H3 OR volatility regime contracts
            if close[i] > camarilla_h3_aligned[i] or volatility_regime_aligned[i] <= 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_1dATR_VolRegime_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0