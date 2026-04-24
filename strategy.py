#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d ATR-based volatility filter and volume spike confirmation.
- Uses 12h timeframe (primary) and 1d HTF for ATR volatility filter (proven SOL/ETH edge).
- Donchian channels calculated from prior 20 periods of 12h high/low.
- Breakout logic: long when price closes above upper Donchian with volume spike and low volatility,
                  short when price closes below lower Donchian with volume spike and low volatility.
- Volatility filter: only trade when 1d ATR(14) < 0.5 * 20-period 1d ATR(14) MA (low vol regime).
- Volume confirmation: current 12h volume > 1.5 * 20-period 12h volume MA.
- Discrete signal size: 0.25 to balance reward and risk, minimizing fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
- Works in both bull/bear: volatility filter avoids choppy markets, Donchian breakouts capture momentum.
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
    
    # Calculate 1d ATR(14) for volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value NaN
    
    atr_14_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_ma_1d = pd.Series(atr_14_1d).rolling(window=20, min_periods=20).mean().values
    low_volatility = atr_14_1d < (0.5 * atr_ma_1d)
    
    # Align 1d data to 12h timeframe
    low_volatility_aligned = align_htf_to_ltf(prices, df_1d, low_volatility)
    
    # Calculate Donchian channels from prior 20 periods of 12h data
    donchian_window = 20
    donchian_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # Shift by 1 to use prior period only (avoid look-ahead)
    donchian_high = np.concatenate([[np.nan], donchian_high[:-1]])
    donchian_low = np.concatenate([[np.nan], donchian_low[:-1]])
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(donchian_window, 20)  # Need Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(low_volatility_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price closes above upper Donchian AND low volatility AND volume spike
            if close[i] > donchian_high[i] and low_volatility_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price closes below lower Donchian AND low volatility AND volume spike
            elif close[i] < donchian_low[i] and low_volatility_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reverts to midpoint of Donchian channels or reverse signal
            donchian_mid = (donchian_high[i] + donchian_low[i]) / 2
            if close[i] <= donchian_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverts to midpoint of Donchian channels or reverse signal
            donchian_mid = (donchian_high[i] + donchian_low[i]) / 2
            if close[i] >= donchian_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dATR_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0