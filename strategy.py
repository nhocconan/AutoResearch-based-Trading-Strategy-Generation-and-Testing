#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d ATR regime filter and volume confirmation.
- Primary timeframe: 12h for execution, HTF: 1d for ATR-based regime detection.
- Entry: Price breaks above 20-period Donchian high (long) or below 20-period Donchian low (short) on 12h close,
         with volume > 1.5x 20-period volume MA, and ATR(14) > ATR(50) (high volatility regime).
- Direction filter: Donchian breakout direction determines trade direction (no opposing trend filter needed).
- ATR regime filter ensures trades occur only during sufficient volatility, reducing whipsaws in low-volatility periods.
- Volume confirmation reduces false breakouts.
- Exit: Price returns to the midpoint of the Donchian channel or ATR-based trailing stop (via signal=0).
- Discrete signal size: 0.25 to balance return and drawdown control.
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
- Works in bull via buying breakouts in high volatility, in bear via selling breakdowns in high volatility.
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
    
    # Calculate 1d ATR(14) and ATR(50) for regime filter
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
    
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_50 = pd.Series(tr).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align ATR values to 12h timeframe (completed 1d bar only)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    atr_50_aligned = align_htf_to_ltf(prices, df_1d, atr_50)
    
    # Regime: high volatility when ATR(14) > ATR(50)
    high_vol_regime = atr_14_aligned > atr_50_aligned
    
    # Calculate 12h Donchian channel (20-period)
    lookback = 20
    donchian_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    donchian_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback, 20, 50)  # Donchian(20), volume MA(20), ATR(50)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(high_vol_regime[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian high with volume spike AND high volatility regime
            if (close[i] > donchian_high[i] and volume_spike[i] and high_vol_regime[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low with volume spike AND high volatility regime
            elif (close[i] < donchian_low[i] and volume_spike[i] and high_vol_regime[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price returns to Donchian midpoint or volatility drops
            if (close[i] < donchian_mid[i] or not high_vol_regime[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price returns to Donchian midpoint or volatility drops
            if (close[i] > donchian_mid[i] or not high_vol_regime[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dATRRegime_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0