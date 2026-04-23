#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d ATR regime filter and volume confirmation.
Uses Donchian channels for breakout detection, 1d ATR ratio for regime filtering (trending vs choppy),
and volume confirmation to avoid false breakouts. Designed for 12h timeframe to capture
medium-term swings in both bull and bear markets. Target: 12-37 trades/year per symbol (50-150 total over 4 years).
Uses discrete position sizing (0.25) to minimize fee churn.
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
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # first bar
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # ATR ratio: short-term / long-term ATR
    # High ratio (>1.2) = expanding volatility (trending)
    # Low ratio (<0.8) = contracting volatility (choppy)
    atr_ratio = atr_14 / atr_50
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Calculate 12h Donchian channels (20-period)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Donchian channels: upper = max(high, 20), lower = min(low, 20)
    donch_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe (previous bar values)
    donch_high_aligned = align_htf_to_ltf(prices, df_12h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_12h, donch_low)
    
    # Calculate volume MA (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # need ATR50 and vol MA20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(atr_ratio_aligned[i]) or np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: ATR ratio > 1.0 = trending market (favor breakouts)
        regime_filter = atr_ratio_aligned[i] > 1.0
        
        # Volume filter: current volume > 1.3x 20-period MA
        vol_filter = volume[i] > 1.3 * vol_ma_20[i]
        
        if position == 0:
            # Long: Break above Donchian upper AND trending regime AND volume confirmation
            if close[i] > donch_high_aligned[i] and regime_filter and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian lower AND trending regime AND volume confirmation
            elif close[i] < donch_low_aligned[i] and regime_filter and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: break of opposite Donchian level
            exit_signal = False
            if position == 1:
                # Exit long on break below Donchian lower
                if close[i] < donch_low_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short on break above Donchian upper
                if close[i] > donch_high_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Donchian20_Breakout_1dATRRegime_VolumeConfirmation"
timeframe = "12h"
leverage = 1.0