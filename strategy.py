#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian channel breakout with 1d ATR regime filter and volume confirmation.
- Primary timeframe: 4h for execution, HTF: 1d for ATR-based regime detection.
- Donchian breakout: Long when price > 20-period high, Short when price < 20-period low.
- Regime filter: Only trade when 1d ATR(14) > 20-period SMA of ATR (high volatility regime).
- Volume confirmation: current volume > 1.5 * 20-period volume MA.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
- Works in bull via buying breakouts in high volatility uptrends, in bear via selling breakdowns in high volatility downtrends.
- ATR regime filter avoids low-volatility choppy markets where breakouts fail.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR(14)
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.maximum(np.abs(low_1d[1:] - close_1d[:-1]), tr1)
    tr = np.concatenate([[np.nan], tr2])  # First TR is undefined
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # 20-period SMA of 1d ATR for regime filter
    atr_ma = pd.Series(atr_14).rolling(window=20, min_periods=20).mean().values
    high_vol_regime = atr_14 > atr_ma  # True when volatility is above average
    
    # Align regime filter to 4h
    high_vol_regime_aligned = align_htf_to_ltf(prices, df_1d, high_vol_regime)
    
    # Donchian channel (20-period) on 4h data
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 20)  # Donchian + volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or
            np.isnan(high_vol_regime_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Only trade in high volatility regime
            if high_vol_regime_aligned[i]:
                if close[i] > highest_20[i] and volume_spike[i]:
                    # Buy on Donchian breakout in high vol regime
                    signals[i] = 0.25
                    position = 1
                elif close[i] < lowest_20[i] and volume_spike[i]:
                    # Sell on Donchian breakdown in high vol regime
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price returns to midpoint of Donchian channel
            midpoint = (highest_20[i] + lowest_20[i]) / 2
            if close[i] < midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to midpoint of Donchian channel
            midpoint = (highest_20[i] + lowest_20[i]) / 2
            if close[i] > midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dATR_Regime_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0