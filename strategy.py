#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ATR filter and volume spike confirmation.
- Uses 4h timeframe (primary) and 1d HTF for ATR-based volatility filter (proven BTC/ETH edge from DB).
- Donchian channels calculated from prior 20-period 4h high/low.
- Breakout logic: long when price closes above upper band with volume spike and ATR expansion,
                  short when price closes below lower band with volume spike and ATR expansion.
- ATR filter: only trade when current 1d ATR > 1.2 * 20-period 1d ATR MA (avoid low volatility chop).
- Volume confirmation: current 4h volume > 1.8 * 20-period 4h volume MA (balanced to avoid overtrading).
- Discrete signal size: 0.25 to balance reward and risk, minimizing fee churn.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
- Works in both bull/bear: ATR expansion filter captures volatility breakouts in all regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 4h Donchian(20) from prior periods
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().shift(1).values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().shift(1).values
    
    # Calculate 1d ATR for volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # First period has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma_1d = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d ATR data to 4h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    atr_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_1d)
    
    # Volume confirmation: current volume > 1.8 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * volume_ma)
    
    # ATR expansion filter: current ATR > 1.2 * ATR MA
    atr_expansion = atr_1d_aligned > (1.2 * atr_ma_1d_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, lookback, 20)  # Need Donchian lookback and sufficient ATR/volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(atr_1d_aligned[i]) or np.isnan(atr_ma_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price closes above upper Donchian band AND volume spike AND ATR expansion
            if close[i] > highest_high[i] and volume_spike[i] and atr_expansion[i]:
                signals[i] = 0.25
                position = 1
            # Short: price closes below lower Donchian band AND volume spike AND ATR expansion
            elif close[i] < lowest_low[i] and volume_spike[i] and atr_expansion[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reverts to middle of Donchian channel or reverse signal
            donchian_mid = (highest_high[i] + lowest_low[i]) / 2
            if close[i] <= donchian_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverts to middle of Donchian channel or reverse signal
            donchian_mid = (highest_high[i] + lowest_low[i]) / 2
            if close[i] >= donchian_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dATR_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0