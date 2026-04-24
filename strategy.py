#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ATR-based volatility filter and volume confirmation.
- Donchian(20) provides clear structure for breakouts in both trending and ranging markets.
- 1d ATR ratio (current ATR(14) / 20-period ATR mean) filters for expanding volatility regimes, effective in bull/bear.
- Volume spike (>1.5x 20-period average) confirms breakout authenticity.
- Discrete position sizing (0.25) limits fee churn while capturing meaningful moves.
- Target trades: 75-200 total over 4 years to avoid fee drag on 4h timeframe.
- Works in bull markets via upward breakouts and bear markets via downward breakdowns with volatility filter.
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
    
    # Get 1d data ONCE before loop for ATR-based volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d ATR ratio: current ATR(14) / 20-period ATR mean (volatility regime filter)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align length
    
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_ma_20 = pd.Series(atr_14).rolling(window=20, min_periods=20).mean().values
    atr_ratio = atr_14 / atr_ma_20
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Donchian(20) channels on 4h
    donchian_h = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_l = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: > 1.5x 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_h[i]) or np.isnan(donchian_l[i]) or 
            np.isnan(atr_ratio_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above Donchian high with volume spike and expanding volatility (ATR ratio > 1.0)
            if close[i] > donchian_h[i] and volume_spike[i] and atr_ratio_aligned[i] > 1.0:
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low with volume spike and expanding volatility (ATR ratio > 1.0)
            elif close[i] < donchian_l[i] and volume_spike[i] and atr_ratio_aligned[i] > 1.0:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below Donchian low OR volatility contracts (ATR ratio < 0.8)
            if close[i] < donchian_l[i] or atr_ratio_aligned[i] < 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above Donchian high OR volatility contracts (ATR ratio < 0.8)
            if close[i] > donchian_h[i] or atr_ratio_aligned[i] < 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_VolatilityFilter_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0