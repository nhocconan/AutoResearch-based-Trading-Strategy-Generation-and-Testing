#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ATR regime filter and volume confirmation.
Long when price breaks above 20-period 4h Donchian high in low volatility regime (ATR ratio < 0.8) with volume spike.
Short when price breaks below 20-period 4h Donchian low in low volatility regime with volume spike.
Uses 1d ATR ratio (ATR(7)/ATR(30)) to filter for low volatility breakouts which have higher success rate.
Designed for 4h timeframe to capture meaningful moves with controlled trade frequency (target: 20-40 trades/year).
Uses discrete position sizing (0.25) to minimize fee drag while maintaining profit potential.
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
    
    # Calculate 4h Donchian(20) channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Donchian channels: 20-period high and low
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 4h timeframe (previous 20-bar completed values)
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Calculate 1d ATR ratio: ATR(7)/ATR(30) for volatility regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    
    # ATR(7) and ATR(30)
    atr_7 = pd.Series(tr).rolling(window=7, min_periods=7).mean().values
    atr_30 = pd.Series(tr).rolling(window=30, min_periods=30).mean().values
    
    # ATR ratio: ATR(7)/ATR(30) - low volatility when ratio < 0.8
    atr_ratio = np.where(atr_30 != 0, atr_7 / atr_30, np.nan)
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Volume spike: current volume > 1.8x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.8 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(30, 20)  # need ATR30 and Donchian20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(atr_ratio_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: low volatility (ATR ratio < 0.8)
        low_volatility = atr_ratio_aligned[i] < 0.8
        
        if position == 0:
            # Long: Break above Donchian high AND low volatility AND volume spike
            if close[i] > donchian_high_aligned[i] and low_volatility and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian low AND low volatility AND volume spike
            elif close[i] < donchian_low_aligned[i] and low_volatility and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: break of opposite Donchian level or volatility expansion
            exit_signal = False
            if position == 1:
                # Exit long on break below Donchian low or volatility expansion
                if close[i] < donchian_low_aligned[i] or atr_ratio_aligned[i] > 1.2:
                    exit_signal = True
            elif position == -1:
                # Exit short on break above Donchian high or volatility expansion
                if close[i] > donchian_high_aligned[i] or atr_ratio_aligned[i] > 1.2:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian20_Breakout_1dATRRegime_VolumeSpike"
timeframe = "4h"
leverage = 1.0