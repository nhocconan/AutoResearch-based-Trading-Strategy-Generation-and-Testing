#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H4/L4 breakout with 1d ATR regime filter and volume spike confirmation.
- Primary timeframe: 4h for execution, HTF: 1d for ATR regime and Camarilla levels.
- Camarilla pivot levels: H4 = close + 1.1*(high-low)/2, L4 = close - 1.1*(high-low)/2.
- Entry: Long when price breaks above H4 with volume spike and ATR(14) > ATR(50) (high volatility regime).
         Short when price breaks below L4 with volume spike and ATR(14) > ATR(50) (high volatility regime).
- Exit: When price reverts to Camarilla Pivot Point (PP) or opposite signal.
- Works in bull via buying breakouts in high vol, in bear via selling breakdowns in high vol.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels and ATR regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) and ATR(50) for regime filter
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_50 = tr.ewm(span=50, adjust=False, min_periods=50).mean().values
    atr_regime = atr_14 > atr_50  # High volatility regime
    
    # Calculate Camarilla levels on 1d
    # H4 = close + 1.1*(high-low)/2
    # L4 = close - 1.1*(high-low)/2
    # PP = (high + low + close) / 3
    camarilla_h4 = df_1d['close'].values + (1.1 * (df_1d['high'].values - df_1d['low'].values) / 2)
    camarilla_l4 = df_1d['close'].values - (1.1 * (df_1d['high'].values - df_1d['low'].values) / 2)
    camarilla_pp = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3
    
    # Align 1d indicators to 4h
    atr_regime_aligned = align_htf_to_ltf(prices, df_1d, atr_regime)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA (on 4h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need enough 1d bars for ATR50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(atr_regime_aligned[i]) or np.isnan(camarilla_h4_aligned[i]) or 
            np.isnan(camarilla_l4_aligned[i]) or np.isnan(camarilla_pp_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Check for Camarilla breakout signals with volume spike and ATR regime
            if volume_spike[i] and atr_regime_aligned[i]:
                # Long: price breaks above H4 in high vol regime
                if close[i] > camarilla_h4_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below L4 in high vol regime
                elif close[i] < camarilla_l4_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price reverts to PP or short signal
            if close[i] < camarilla_pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverts to PP or long signal
            if close[i] > camarilla_pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H4L4_Breakout_1dATRRegime_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0