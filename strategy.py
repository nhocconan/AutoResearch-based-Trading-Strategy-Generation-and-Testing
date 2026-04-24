#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ATR regime filter and volume confirmation.
- Primary timeframe: 4h for execution, HTF: 1d for ATR-based regime detection.
- Entry: Price breaks above Donchian upper (long) or below lower (short) on 4h close, with volume > 1.5x 20-period volume MA.
- Regime filter: Only trade when 1d ATR(14) > 1.5 * ATR(50) (high volatility regime) to avoid choppy markets.
- Exit: Price returns to Donchian midpoint or opposite band touch.
- Discrete signal size: 0.25 to balance return and drawdown control.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
- Works in bull via buying breakouts in uptrend, in bear via selling breakdowns in downtrend, and avoids ranging markets via volatility filter.
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
    
    # Calculate 1d ATR(14) and ATR(50) for volatility regime filter
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
    tr1[0] = high_1d[0] - low_1d[0]  # first bar
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) and ATR(50)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # Volatility regime: ATR(14) > 1.5 * ATR(50) indicates high volatility/trending market
    volatility_regime = atr_14 > (1.5 * atr_50)
    
    # Align volatility regime to 4h timeframe (completed 1d bar only)
    volatility_regime_aligned = align_htf_to_ltf(prices, df_1d, volatility_regime)
    
    # Calculate 4h Donchian channels (20-period)
    # Use rolling window on 4h data directly
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_ma
    donchian_lower = low_ma
    donchian_mid = (donchian_upper + donchian_lower) / 2
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20) + 1  # Need ATR(50), Donchian(20), volume MA(20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(volatility_regime_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian upper with volume spike AND high volatility regime
            if (close[i] > donchian_upper[i] and volume_spike[i] and volatility_regime_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian lower with volume spike AND high volatility regime
            elif (close[i] < donchian_lower[i] and volume_spike[i] and volatility_regime_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price returns to Donchian midpoint or touches lower band
            if (close[i] < donchian_mid[i] or close[i] < donchian_lower[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price returns to Donchian midpoint or touches upper band
            if (close[i] > donchian_mid[i] or close[i] > donchian_upper[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_ATRRegime_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0