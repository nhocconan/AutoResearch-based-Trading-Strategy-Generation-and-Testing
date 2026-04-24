#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ADX regime filter and volume confirmation.
- Primary timeframe: 4h for execution, HTF: 1d for ADX regime and Donchian levels.
- Donchian channels calculated from previous 20 4h bars (using 1d alignment).
- ADX(14) on 1d: >25 = trending (follow breakout), <20 = ranging (fade breakout).
- Volume confirmation: current 4h volume > 1.5 * 20-period volume MA.
- Entry: Long when price breaks above Donchian upper + volume spike + ADX>25.
         Short when price breaks below Donchian lower + volume spike + ADX>25.
         In ranging markets (ADX<20): fade at Donchian bands with volume confirmation.
- Exit: Opposite Donchian band touch or volume spike reversal.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)"""
    if len(high) < period + 1:
        return np.full_like(high, np.nan, dtype=float)
    
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                       np.maximum(high[1:] - high[:-1], 0), 0)
    dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                        np.maximum(low[:-1] - low[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    atr = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=period, adjust=False, min_periods=period).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    return adx

def calculate_donchian(high, low, period=20):
    """Calculate Donchian channels"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need enough for ADX calculation
        return np.zeros(n)
    
    # Calculate 1d ADX for regime filter
    adx_1d = calculate_adx(
        df_1d['high'].values,
        df_1d['low'].values,
        df_1d['close'].values,
        period=14
    )
    
    # Align 1d ADX to 4h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 4h Donchian channels (using 20-period)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA (on 4h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # Need enough 1d bars for ADX and 4h bars for Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Check for breakout or fade signals based on regime
            if volume_spike[i]:
                # Trending regime (ADX > 25): follow breakout
                if adx_aligned[i] > 25:
                    if close[i] > donchian_upper[i]:
                        signals[i] = 0.25
                        position = 1
                    elif close[i] < donchian_lower[i]:
                        signals[i] = -0.25
                        position = -1
                # Ranging regime (ADX < 20): fade at bands
                elif adx_aligned[i] < 20:
                    if close[i] >= donchian_upper[i]:
                        signals[i] = -0.25  # Short at upper band
                        position = -1
                    elif close[i] <= donchian_lower[i]:
                        signals[i] = 0.25   # Long at lower band
                        position = 1
        elif position == 1:
            # Long exit: price touches lower Donchian band or volume spike reversal
            if close[i] <= donchian_lower[i] or (volume_spike[i] and close[i] < close[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price touches upper Donchian band or volume spike reversal
            if close[i] >= donchian_upper[i] or (volume_spike[i] and close[i] > close[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_1dADX_Regime_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0