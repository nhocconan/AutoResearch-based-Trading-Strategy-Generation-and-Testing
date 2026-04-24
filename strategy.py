#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with 1d ATR regime filter and volume confirmation.
- Primary timeframe: 6h for execution, HTF: 1d for ATR regime (adaptive volatility filter).
- Donchian channel: 20-period high/low from 6h data.
- Breakout: Close > upper band (long) or Close < lower band (short).
- Regime filter: Only trade when 1d ATR(14) > 0.6 * 20-period SMA of ATR(14) (high volatility regime).
- Volume confirmation: Volume > 1.5 * 20-period volume MA.
- Works in bull via buying breakouts in high vol, in bear via selling breakdowns in high vol.
- Avoids low-volatility choppy markets where breakouts fail.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 75-200 total trades over 4 years (19-50/year) for 6h timeframe.
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
    
    # Get 1d data for ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) for regime filter
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 20-period SMA of ATR(14) for regime threshold
    atr_ma = pd.Series(atr_14).rolling(window=20, min_periods=20).mean().values
    atr_regime = atr_14 > (0.6 * atr_ma)  # High volatility regime
    
    # Align 1d ATR regime to 6h
    atr_regime_aligned = align_htf_to_ltf(prices, df_1d, atr_regime)
    
    # Calculate 6h Donchian(20) channels
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_roll
    donchian_lower = low_roll
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 20)  # Donchian + volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(atr_regime_aligned[i]) or np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Check for Donchian breakout with volume confirmation and ATR regime
            if volume_confirm[i] and atr_regime_aligned[i]:
                # Long breakout: close > upper Donchian band
                if close[i] > donchian_upper[i]:
                    signals[i] = 0.25
                    position = 1
                # Short breakdown: close < lower Donchian band
                elif close[i] < donchian_lower[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price re-enters Donchian channel or opposite signal
            if close[i] < donchian_lower[i]:  # Exit when price falls below lower band
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price re-enters Donchian channel or opposite signal
            if close[i] > donchian_upper[i]:  # Exit when price rises above upper band
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_Breakout_1dATRRegime_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0