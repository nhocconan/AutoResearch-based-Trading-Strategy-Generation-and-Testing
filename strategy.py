#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla H4/L4 breakout with 1d ATR regime filter and volume confirmation.
- Camarilla H4/L4 levels from 1d chart identify key intraday support/resistance with high probability reactions.
- 1d ATR regime filter: only trade when ATR(14) > 1.5 * ATR(50) to ensure sufficient volatility for breakouts.
- Volume confirmation (>2.0x 24-period average) validates breakout strength and reduces false signals.
- Discrete position sizing (0.25) balances return potential with fee drag minimization.
- Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe to avoid excessive fees.
- Works in bull/bear markets via volatility regime filter and volume confirmation.
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
    
    # Get 1d data ONCE before loop for Camarilla levels and ATR regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla levels on 1d (using previous completed 1d bar)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: H4 = close + 1.1/2 * (high - low), L4 = close - 1.1/2 * (high - low)
    camarilla_h4 = close_1d + 1.1/2 * (high_1d - low_1d)
    camarilla_l4 = close_1d - 1.1/2 * (high_1d - low_1d)
    
    # Align Camarilla levels to 6h timeframe (using previous completed 1d bar)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # 1d ATR regime filter: ATR(14) > 1.5 * ATR(50) indicates high volatility regime
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.maximum(np.abs(low_1d[1:] - close_1d[:-1]), tr1)
    tr = np.concatenate([[np.nan], tr2])  # first TR is NaN
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    atr_regime = atr_14 > 1.5 * atr_50
    
    # Align ATR regime to 6h timeframe
    atr_regime_aligned = align_htf_to_ltf(prices, df_1d, atr_regime.astype(float))
    
    # Volume confirmation: > 2.0x 24-period average volume (6h * 4 = 1 day)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > 2.0 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(24, 50) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(atr_regime_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above Camarilla H4 with volume spike and high volatility regime
            if close[i] > camarilla_h4_aligned[i] and volume_spike[i] and atr_regime_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below Camarilla L4 with volume spike and high volatility regime
            elif close[i] < camarilla_l4_aligned[i] and volume_spike[i] and atr_regime_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below Camarilla L4 OR volatility regime ends
            if close[i] < camarilla_l4_aligned[i] or not atr_regime_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above Camarilla H4 OR volatility regime ends
            if close[i] > camarilla_h4_aligned[i] or not atr_regime_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_H4L4_1dATRRegime_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0