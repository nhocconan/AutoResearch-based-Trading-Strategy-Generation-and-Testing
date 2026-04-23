#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with 1d ATR regime filter and volume confirmation.
Uses discrete position sizing (0.25) to minimize fee churn. Works in bull/bear via ATR-based regime detection:
- High ATR regime (ATR(14) > ATR(50)): trend-following breakouts
- Low ATR regime (ATR(14) <= ATR(50)): mean-reversion at Donchian bounds
Volume confirmation reduces false breakouts. Target: 12-37 trades/year per symbol.
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
    tr1[0] = high_1d[0] - low_1d[0]  # first bar has no previous close
    tr2[0] = high_1d[0] - close_1d[0]  # approximate
    tr3[0] = low_1d[0] - close_1d[0]   # approximate
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    atr_50_aligned = align_htf_to_ltf(prices, df_1d, atr_50)
    
    # Calculate 6h Donchian(20) channels
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    # Donchian channels: upper = max(high, lookback=20), lower = min(low, lookback=20)
    donch_upper = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    donch_lower = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    donch_upper_aligned = align_htf_to_ltf(prices, df_6h, donch_upper)
    donch_lower_aligned = align_htf_to_ltf(prices, df_6h, donch_lower)
    
    # Calculate volume MA (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # need ATR50, volume MA20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(atr_14_aligned[i]) or np.isnan(atr_50_aligned[i]) or 
            np.isnan(donch_upper_aligned[i]) or np.isnan(donch_lower_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # ATR regime: high volatility = trend following, low volatility = mean reversion
        high_vol_regime = atr_14_aligned[i] > atr_50_aligned[i]
        
        # Volume filter: current volume > 1.5x 20-period MA
        vol_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            if high_vol_regime:
                # Trend following: breakout in direction of break
                if close[i] > donch_upper_aligned[i] and vol_filter:
                    signals[i] = 0.25
                    position = 1
                elif close[i] < donch_lower_aligned[i] and vol_filter:
                    signals[i] = -0.25
                    position = -1
            else:
                # Mean reversion: fade at Donchian bounds
                if close[i] < donch_lower_aligned[i] and vol_filter:
                    signals[i] = 0.25  # long at lower bound
                    position = 1
                elif close[i] > donch_upper_aligned[i] and vol_filter:
                    signals[i] = -0.25  # short at upper bound
                    position = -1
        else:
            # Exit conditions
            exit_signal = False
            if position == 1:
                # Exit long: price reaches upper Donchian (trend) or lower Donchian (mean reversion)
                if close[i] >= donch_upper_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short: price reaches lower Donchian (trend) or upper Donchian (mean reversion)
                if close[i] <= donch_lower_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Donchian20_ATRRegime_VolumeFilter"
timeframe = "6h"
leverage = 1.0