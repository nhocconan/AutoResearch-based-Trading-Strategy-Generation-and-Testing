#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with 1d ATR regime filter and volume confirmation.
- Primary timeframe: 6h for execution, HTF: 1d for ATR regime detection.
- Entry: Price breaks above Donchian(20) upper band (long) or below lower band (short) on 6h close, with volume > 1.5x 20-period volume MA.
- Regime filter: only trade when 1d ATR(14) is above its 50-period MA (high volatility regime) to avoid choppy markets.
- Donchian breakouts capture strong momentum moves; ATR filter ensures we only trade in volatile, trending conditions.
- Volume confirmation reduces false breakouts.
- Exit: Price returns to the midpoint of the Donchian channel or regime filter reversal.
- Discrete signal size: 0.25 to balance return and drawdown control.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
- Works in bull via buying breakouts in uptrend, in bear via selling breakdowns in downtrend, and avoids ranging markets via ATR filter.
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
    
    # Calculate 1d ATR(14) for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # first period
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    # ATR(50) MA for regime
    atr_ma_50 = pd.Series(atr_14).rolling(window=50, min_periods=50).mean().values
    # Regime: high volatility when ATR(14) > ATR(50) MA
    atr_regime = atr_14 > atr_ma_50
    atr_regime_aligned = align_htf_to_ltf(prices, df_1d, atr_regime)
    
    # Donchian(20) channels on 6h data
    donchian_window = 20
    donchian_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_high).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_low).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, donchian_window, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(atr_regime_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian high with volume spike AND high volatility regime
            if (close[i] > donchian_high[i] and volume_spike[i] and atr_regime_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low with volume spike AND high volatility regime
            elif (close[i] < donchian_low[i] and volume_spike[i] and atr_regime_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price returns to Donchian midpoint or regime shifts to low volatility
            if (close[i] < donchian_mid[i] or not atr_regime_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price returns to Donchian midpoint or regime shifts to low volatility
            if (close[i] > donchian_mid[i] or not atr_regime_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_1dATRRegime_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0