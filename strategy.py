#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout + 1d ATR regime filter + volume expansion.
- Primary timeframe: 12h for execution, HTF: 1d for ATR-based regime detection and trend context.
- Entry: 12h close breaks above 20-period Donchian high OR below 20-period Donchian low
         with volume > 1.5x 20-period volume MA AND ATR regime expansion (current ATR > 1.2x 20-period ATR MA).
- Direction: Donchian breakout direction determines signal (long for upper break, short for lower break).
- Regime filter: Only trade when volatility is expanding (avoid choppy/low-vol environments).
- Exit: Opposite Donchian level touch (long exits at lower Donchian, short exits at upper Donchian).
- Discrete signal size: 0.25 to balance return and drawdown control.
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
- Works in bull via breakout continuation, in bear via volatility expansion capturing panic moves.
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
    
    # Calculate 1d ATR for regime filter (using true range)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True range for 1d: max(high-low, |high-close_prev|, |low-close_prev|)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Set first TR to high-low (no previous close)
    tr[0] = tr1[0]
    
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_ma_1d = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    atr_expansion = atr_1d > (1.2 * atr_ma_1d)  # Volatility expanding
    
    # Align 1d ATR expansion to 12h timeframe
    atr_expansion_aligned = align_htf_to_ltf(prices, df_1d, atr_expansion)
    
    # Calculate 20-period Donchian channels on 12h data
    donchian_window = 20
    donchian_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_expansion = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(donchian_window, 20)  # Need Donchian, volume MA, ATR MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_expansion[i]) or np.isnan(atr_expansion_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high with volume expansion AND ATR regime expansion
            if close[i] > donchian_high[i] and volume_expansion[i] and atr_expansion_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with volume expansion AND ATR regime expansion
            elif close[i] < donchian_low[i] and volume_expansion[i] and atr_expansion_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price returns below Donchian low (mean reversion) or volatility contraction
            if close[i] < donchian_low[i] or not atr_expansion_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns above Donchian high (mean reversion) or volatility contraction
            if close[i] > donchian_high[i] or not atr_expansion_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dATRRegime_VolumeExpansion_v1"
timeframe = "12h"
leverage = 1.0