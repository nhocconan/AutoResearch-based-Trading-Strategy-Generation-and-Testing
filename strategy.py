#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d ATR volatility filter and volume confirmation.
- Primary timeframe: 12h for execution, HTF: 1d for ATR regime filter.
- Donchian(20) on 12h: Upper = 20-bar high, Lower = 20-bar low.
- ATR filter: Only trade when 1d ATR(14) > 20-bar SMA of ATR(14) (high volatility regime).
- Volume confirmation: 12h volume > 1.5 * 20-bar volume SMA.
- Entry: Long when close > Donchian Upper + volume confirmation + ATR filter.
         Short when close < Donchian Lower + volume confirmation + ATR filter.
- Exit: Opposite Donchian breakout or ATR filter fails.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
- Works in both bull (breakouts catch trends) and bear (volatility filters avoid false breakouts in low vol).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ATR(14)
    # True Range
    tr1 = pd.Series(df_1d['high']).diff().abs()
    tr2 = (pd.Series(df_1d['high']) - pd.Series(df_1d['low'].shift())).abs()
    tr3 = (pd.Series(df_1d['low']) - pd.Series(df_1d['close'].shift())).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # ATR filter: current ATR > 20-period SMA of ATR (high volatility regime)
    atr_ma = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    atr_filter = atr > atr_ma
    
    # Align ATR filter to 12h
    atr_filter_aligned = align_htf_to_ltf(prices, df_1d, atr_filter)
    
    # Donchian(20) on 12h
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_roll
    donchian_lower = low_roll
    
    # Volume confirmation on 12h
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(30, 20)  # Need enough bars for Donchian and ATR
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(atr_filter_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        atr_ok = atr_filter_aligned[i]
        vol_ok = volume_spike[i]
        
        if position == 0:
            # Check for entry signals with filters
            if vol_ok and atr_ok:
                if close[i] > donchian_upper[i]:
                    signals[i] = 0.25
                    position = 1
                elif close[i] < donchian_lower[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: close < Donchian Lower or filters fail
            if close[i] < donchian_lower[i] or not (atr_ok and vol_ok):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: close > Donchian Upper or filters fail
            if close[i] > donchian_upper[i] or not (atr_ok and vol_ok):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dATRFilter_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0