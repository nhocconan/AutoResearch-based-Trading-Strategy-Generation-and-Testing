#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d ATR regime filter and volume confirmation.
Long when price breaks above 12h Donchian upper band (20-period) AND 1d ATR ratio > 0.8 (low volatility regime) AND volume > 1.5x 20-period MA.
Short when price breaks below 12h Donchian lower band (20-period) AND 1d ATR ratio > 0.8 AND volume > 1.5x 20-period MA.
Exit when price returns to the midpoint of the Donchian channel or opposite breakout occurs.
Uses ATR regime filter to avoid whipsaws in high volatility and focus on breakouts in stable conditions.
Designed for ~15-25 trades/year with proven edge from DB top performers.
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
    
    # Calculate 12h Donchian channel (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_roll
    donchian_lower = low_roll
    donchian_mid = (donchian_upper + donchian_lower) / 2
    
    # Calculate 1d ATR for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = 0  # First period has no previous close
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) calculation
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # ATR ratio: current ATR / 50-period ATR mean (regime filter)
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr_14 / atr_50
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Volume MA (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50, 20)  # need Donchian20, ATR50, volume MA20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(atr_ratio_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: ATR ratio > 0.8 (low volatility regime)
        vol_regime = atr_ratio_aligned[i] > 0.8
        
        # Volume filter: 12h volume > 1.5x 20-period MA
        vol_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        # Donchian breakout conditions
        breakout_up = close[i] > donchian_upper[i]  # Break above upper band
        breakout_down = close[i] < donchian_lower[i]  # Break below lower band
        return_to_mid = abs(close[i] - donchian_mid[i]) < 0.1 * abs(donchian_upper[i] - donchian_lower[i])  # Return to midpoint
        opposite_extreme = (position == 1 and breakout_down) or \
                           (position == -1 and breakout_up)
        
        if position == 0:
            # Long: Break above upper band AND low volatility regime AND volume confirmation
            if breakout_up and vol_regime and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Break below lower band AND low volatility regime AND volume confirmation
            elif breakout_down and vol_regime and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: return to midpoint or opposite extreme hit
            exit_signal = False
            if position == 1:
                exit_signal = return_to_mid or opposite_extreme
            elif position == -1:
                exit_signal = return_to_mid or opposite_extreme
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Donchian20_Breakout_1dATR_Regime_VolumeConfirmation"
timeframe = "12h"
leverage = 1.0