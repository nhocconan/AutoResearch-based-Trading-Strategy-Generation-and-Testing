#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ATR regime filter and volume confirmation.
Long when price breaks above Donchian upper band in low volatility regime (ATR ratio < 0.8).
Short when price breaks below Donchian lower band in low volatility regime.
Volume confirmation requires current volume > 1.5x 20-period MA.
Uses discrete position sizing (0.25) to limit fee drift. Designed for 4h timeframe
to capture breakouts with controlled trade frequency (~20-50 trades/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d ATR(14) for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # first bar
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate 20-period ATR for volatility ratio (using 1d ATR)
    atr_ma_20_1d = pd.Series(atr_14_1d).rolling(window=20, min_periods=20).mean().values
    atr_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_20_1d)
    
    # ATR ratio: current ATR / 20-period MA ATR (< 0.8 = low volatility regime)
    atr_ratio = atr_14_1d_aligned / atr_ma_20_1d_aligned
    low_vol_regime = atr_ratio < 0.8
    
    # Calculate Donchian channels (20-period) on 4h data
    donchian_window = 20
    donchian_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # Calculate volume spike: current volume > 1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.5 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(donchian_window, 20, 14+20)  # Donchian20, volMA20, ATR14+MA20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(atr_ratio[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian high AND low vol regime AND volume spike
            if close[i] > donchian_high[i] and low_vol_regime[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low AND low vol regime AND volume spike
            elif close[i] < donchian_low[i] and low_vol_regime[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price breaks opposite Donchian level
            exit_signal = False
            if position == 1:
                # Exit long on break below Donchian low
                if close[i] < donchian_low[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short on break above Donchian high
                if close[i] > donchian_high[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian20_Breakout_1dATRRegime_VolumeConfirmation"
timeframe = "4h"
leverage = 1.0