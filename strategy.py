#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ATR regime filter and volume confirmation.
- Uses 4h timeframe (primary) and 1d HTF for ATR-based regime classification (proven edge from DB).
- Donchian channels calculated from prior 20-period 4h high/low.
- Breakout logic: long when price closes above upper band with volume expansion and low volatility regime,
                  short when price closes below lower band with volume expansion and low volatility regime.
- Regime filter: only trade when 1d ATR(14) / 1d ATR(50) < 1.2 (low volatility environment favors breakouts).
- Volume confirmation: current 4h volume > 1.5 * 20-period 4h volume MA (moderate to avoid overtrading).
- Discrete signal size: 0.25 to balance reward and risk, minimizing fee churn.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
- Works in both bull/bear: volatility regime filter avoids choppy markets, Donchian breakouts capture momentum.
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
    
    # Calculate 1d ATR for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # ATR(14) and ATR(50)
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_50 = pd.Series(tr).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Regime: low volatility when ATR(14)/ATR(50) < 1.2
    volatility_ratio = atr_14 / atr_50
    low_vol_regime = volatility_ratio < 1.2
    
    # Align regime to 4h timeframe
    low_vol_regime_aligned = align_htf_to_ltf(prices, df_1d, low_vol_regime)
    
    # Donchian(20) from prior 20 periods (avoid look-ahead)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_expansion = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 50, 20)  # Need sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(low_vol_regime_aligned[i]) or np.isnan(volume_expansion[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price closes above upper Donchian band AND low vol regime AND volume expansion
            if close[i] > donchian_high[i] and low_vol_regime_aligned[i] and volume_expansion[i]:
                signals[i] = 0.25
                position = 1
            # Short: price closes below lower Donchian band AND low vol regime AND volume expansion
            elif close[i] < donchian_low[i] and low_vol_regime_aligned[i] and volume_expansion[i]:
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

name = "4h_Donchian20_1dATRRegime_VolumeExpansion_v1"
timeframe = "4h"
leverage = 1.0