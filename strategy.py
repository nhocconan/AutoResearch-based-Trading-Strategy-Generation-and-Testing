#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with weekly ATR regime filter and volume confirmation.
Long when price breaks above 20-period Donchian high AND weekly ATR(14) > weekly ATR(50) (expanding volatility) AND volume > 2.0x 20-period average.
Short when price breaks below 20-period Donchian low AND weekly ATR(14) > weekly ATR(50) AND volume > 2.0x 20-period average.
Exit when price touches the opposite Donchian level or weekly ATR regime contracts (ATR14 < ATR50).
Uses 1w HTF for ATR regime to avoid whipsaws in low-volatility environments. Target: 50-150 total trades over 4 years (12-37/year).
Weekly ATR expansion indicates institutional participation and reduces false breakouts.
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
    
    # Calculate 20-period Donchian channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-period volume average for spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate weekly ATR regime filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range calculation
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First TR is undefined
    
    # ATR(14) and ATR(50)
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_50 = pd.Series(tr).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Weekly ATR regime: expanding volatility (ATR14 > ATR50)
    atr_regime_expanding = atr_14 > atr_50
    
    # Align weekly ATR regime to 6h timeframe
    atr_regime_aligned = align_htf_to_ltf(prices, df_1w, atr_regime_expanding.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50)  # Donchian (20), volume MA (20), ATR50 (50)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr_regime_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_regime = bool(atr_regime_aligned[i])  # True if expanding
        
        if position == 0:
            # Long: Break above Donchian high AND expanding volatility AND volume spike
            if price > donchian_high[i] and atr_regime and volume[i] > 2.0 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian low AND expanding volatility AND volume spike
            elif price < donchian_low[i] and atr_regime and volume[i] > 2.0 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: price touches Donchian low OR volatility contracts
                if price < donchian_low[i] or not atr_regime:
                    exit_signal = True
            elif position == -1:
                # Short exit: price touches Donchian high OR volatility contracts
                if price > donchian_high[i] or not atr_regime:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Donchian20_Breakout_1wATRRegime_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0