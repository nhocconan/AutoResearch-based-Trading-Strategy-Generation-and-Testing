#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ATR regime filter and volume confirmation.
Donchian channels identify volatility-based breakouts. ATR regime filter (ATR(7)/ATR(30) > 1.5) 
identifies high volatility environments where breakouts are more likely to succeed. 
Volume confirmation avoids low-conviction signals. Designed for 4h timeframe to balance 
trade frequency and capture medium-term moves in both bull/bear markets. 
Target: 20-50 trades/year per symbol (80-200 total over 4 years).
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
    
    # Calculate 1d ATR(30) and ATR(7) for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for 1d
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # first bar
    tr2[0] = np.abs(high_1d[0] - close_1d[0])  # first bar
    tr3[0] = np.abs(low_1d[0] - close_1d[0])   # first bar
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_30_1d = pd.Series(tr).rolling(window=30, min_periods=30).mean().values
    atr_7_1d = pd.Series(tr).rolling(window=7, min_periods=7).mean().values
    
    # ATR regime: high volatility when short ATR > long ATR * 1.5
    atr_ratio = atr_7_1d / (atr_30_1d + 1e-10)
    vol_regime = atr_ratio > 1.5
    vol_regime_aligned = align_htf_to_ltf(prices, df_1d, vol_regime)
    
    # Calculate 4h Donchian(20) channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Donchian channels based on previous 20 bars
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Align Donchian levels to 4h timeframe (previous bar values)
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Calculate volume MA (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(30, 20)  # need ATR30 and vol MA20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(vol_regime_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: 4h volume > 1.3x 20-period MA
        vol_filter = volume[i] > 1.3 * vol_ma_20[i]
        
        # Volatility regime filter
        regime_filter = vol_regime_aligned[i]
        
        if position == 0:
            # Long: Break above Donchian high AND volume confirmation AND high vol regime
            if close[i] > donchian_high_aligned[i] and vol_filter and regime_filter:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian low AND volume confirmation AND high vol regime
            elif close[i] < donchian_low_aligned[i] and vol_filter and regime_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: break of opposite Donchian level
            exit_signal = False
            if position == 1:
                # Exit long on break below Donchian low
                if close[i] < donchian_low_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short on break above Donchian high
                if close[i] > donchian_high_aligned[i]:
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