#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with 1d ATR regime filter and volume confirmation.
- Long when price breaks above Donchian upper band (20-period high) AND ATR(14) > ATR(50) (high volatility regime) AND volume > 1.5 * median volume
- Short when price breaks below Donchian lower band (20-period low) AND ATR(14) > ATR(50) (high volatility regime) AND volume > 1.5 * median volume
- Exit when price crosses the Donchian middle band (20-period midpoint) OR volatility drops (ATR(14) < ATR(50))
- Uses 6h primary timeframe with 1d HTF for ATR regime to avoid whipsaws in low volatility
- Donchian breakouts capture momentum bursts; ATR regime ensures trades only in high conviction moves
- Volume confirmation filters out low-participation fakeouts
- Designed for BTC/ETH with edge in volatile breakout regimes (works in both bull and bear markets when volatility expands)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period)
    def rolling_max(arr, window):
        return pd.Series(arr).rolling(window=window, min_periods=window).max().values
    def rolling_min(arr, window):
        return pd.Series(arr).rolling(window=window, min_periods=window).min().values
    
    donchian_high = rolling_max(high, 20)
    donchian_low = rolling_min(low, 20)
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Get 1d data ONCE before loop for ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) and ATR(50) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR calculations
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_50 = pd.Series(tr).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d ATR values to 6h timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    atr_50_aligned = align_htf_to_ltf(prices, df_1d, atr_50)
    
    # Volume confirmation: volume > 1.5 * median volume of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    volume_confirm = volume > (1.5 * vol_median)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(donchian_mid[i]) or
            np.isnan(atr_14_aligned[i]) or np.isnan(atr_50_aligned[i]) or np.isnan(vol_median[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility regime: only trade when ATR(14) > ATR(50) (expanding volatility)
        high_volatility = atr_14_aligned[i] > atr_50_aligned[i]
        
        if position == 0:
            # Long: break above upper band, high volatility, volume confirmation
            if close[i] > donchian_high[i] and high_volatility and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below lower band, high volatility, volume confirmation
            elif close[i] < donchian_low[i] and high_volatility and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: cross below middle band OR volatility contraction
            if close[i] < donchian_mid[i] or not high_volatility:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: cross above middle band OR volatility contraction
            if close[i] > donchian_mid[i] or not high_volatility:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_1dATRRegime_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0