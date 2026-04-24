#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ATR regime filter and volume confirmation.
- Primary timeframe: 4h for execution, HTF: 1d for ATR regime and trend filter.
- Donchian channels calculated from 20-period 4h high/low.
- Entry: Long when price breaks above upper Donchian with volume spike and ATR ratio > 0.8 (low volatility regime).
         Short when price breaks below lower Donchian with volume spike and ATR ratio > 0.8.
- Exit: When price returns to the midpoint of the Donchian channel (mean reversion edge).
- Works in bull via buying breakouts in low volatility, in bear via selling breakdowns in low volatility.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
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
    
    # Get 1d data for ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) for volatility regime
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with indices
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d ATR ratio (current ATR / 50-period ATR) for regime filter
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    atr_ratio = np.where(atr_50 > 0, atr_14 / atr_50, 1.0)
    
    # Calculate 4h Donchian channels (20-period)
    donchian_window = 20
    upper_donchian = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    lower_donchian = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    middle_donchian = (upper_donchian + lower_donchian) / 2.0
    
    # Align 1d ATR ratio to 4h
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA (on 4h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(donchian_window, 20, 50)  # Need enough bars for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(upper_donchian[i]) or np.isnan(lower_donchian[i]) or 
            np.isnan(middle_donchian[i]) or np.isnan(atr_ratio_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Check for breakout signals with volume spike and low volatility regime
            if volume_spike[i] and atr_ratio_aligned[i] > 0.8:  # Low volatility regime (ATR ratio > 0.8)
                # Bullish breakout: price > upper Donchian
                if close[i] > upper_donchian[i]:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakdown: price < lower Donchian
                elif close[i] < lower_donchian[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price returns to middle Donchian (mean reversion)
            if close[i] <= middle_donchian[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to middle Donchian (mean reversion)
            if close[i] >= middle_donchian[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_1dATRRegime_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0