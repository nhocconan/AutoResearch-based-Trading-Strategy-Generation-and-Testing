#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d ATR filter and volume confirmation.
- Primary timeframe: 12h, HTF: 1d for ATR-based volatility regime filter
- Donchian channels calculated from prior 12h OHLC: upper = max(high, 20), lower = min(low, 20)
- Breakout logic: long when price crosses above upper band with volume confirmation, short when price crosses below lower band
- Volatility filter: only trade when 1d ATR(14) > 20-period 1d ATR MA (avoid low-volatility chop)
- Volume confirmation: current 12h volume > 1.5 * 20-period 12h volume MA
- Exit: reverse signal or when price reverts to prior 12h close (mean reversion)
- Discrete signal size: 0.25 to balance return and risk
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe as per research
- Works in both bull/bear: volatility filter avoids chop, Donchian breakouts capture momentum in all regimes
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d ATR(14) for volatility regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # First bar: no previous close
    tr2[0] = high_1d[0] - close_1d[0]  # Approximation for first bar
    tr3[0] = close_1d[0] - low_1d[0]   # Approximation for first bar
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma_20 = pd.Series(atr_14).rolling(window=20, min_periods=20).mean().values
    volatility_filter = atr_14 > atr_ma_20  # High volatility regime
    
    volatility_filter_aligned = align_htf_to_ltf(prices, df_1d, volatility_filter)
    
    # Calculate prior 12h Donchian channels (20-period)
    # Need to resample to 12h first using mtf_data
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Donchian channels: 20-period high/low
    donchian_upper = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    donchian_upper_aligned = align_htf_to_ltf(prices, df_12h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_12h, donchian_lower)
    
    # Volume confirmation: current 12h volume > 1.5 * 20-period 12h volume MA
    volume_12h = df_12h['volume'].values
    volume_ma_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_confirm_12h = volume_12h > (1.5 * volume_ma_20)
    volume_confirm_aligned = align_htf_to_ltf(prices, df_12h, volume_confirm_12h)
    
    # Mean reversion exit: price reverts to prior 12h close
    prev_close_12h = df_12h['close'].shift(1).values
    prev_close_aligned = align_htf_to_ltf(prices, df_12h, prev_close_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # Need ATR(14)+MA(20) and Donchian(20) and volume MA(20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(volatility_filter_aligned[i]) or np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or np.isnan(volume_confirm_aligned[i]) or
            np.isnan(prev_close_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price crosses above upper band AND volatility filter AND volume confirmation
            if close[i] > donchian_upper_aligned[i] and close[i-1] <= donchian_upper_aligned[i-1] and volatility_filter_aligned[i] and volume_confirm_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below lower band AND volatility filter AND volume confirmation
            elif close[i] < donchian_lower_aligned[i] and close[i-1] >= donchian_lower_aligned[i-1] and volatility_filter_aligned[i] and volume_confirm_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reverts to prior 12h close (mean reversion) or reverse signal
            if close[i] <= prev_close_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverts to prior 12h close (mean reversion) or reverse signal
            if close[i] >= prev_close_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dATR_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0