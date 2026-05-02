#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ATR regime filter and volume confirmation
# Uses Donchian channel for structure, 1d ATR ratio to filter choppy/low volatility regimes
# Volume spike ensures institutional participation and reduces false breakouts
# Discrete position sizing 0.25 balances risk and minimizes fee churn
# Targets 20-50 trades/year (80-200 total over 4 years) to stay within fee drag limits
# Works in both bull and bear markets by only taking breakouts with volume confirmation in favorable volatility regimes
# ATR regime filter avoids whipsaws in ranging markets while capturing trending moves

name = "4h_Donchian20_1dATR_Ratio_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) and its 30-period average for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with close_1d index
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma_30 = pd.Series(atr_14).rolling(window=30, min_periods=30).mean().values
    atr_ratio = atr_14 / atr_ma_30  # >1 = expanding volatility (trending), <1 = contracting volatility (choppy)
    
    # Align ATR ratio to 4h
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Calculate 4h Donchian(20) channels
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Calculate 4h volume confirmation (1.8x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Donchian and ATR ratio)
    start_idx = 50  # max(20 for Donchian, 30 for ATR MA) + buffer
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(high_ma[i]) or np.isnan(low_ma[i]) or 
            np.isnan(atr_ratio_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade when volatility is expanding or neutral (avoid choppy markets)
        # ATR ratio > 0.9 allows trading in slightly contracting to expanding volatility
        volatility_filter = atr_ratio_aligned[i] > 0.9
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above upper Donchian AND volume confirm AND volatility filter
            if (close[i] > high_ma[i] and 
                volume_confirm[i] and 
                volatility_filter):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian AND volume confirm AND volatility filter
            elif (close[i] < low_ma[i] and 
                  volume_confirm[i] and 
                  volatility_filter):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price breaks below lower Donchian OR volatility contracts significantly
            if (close[i] < low_ma[i] or 
                atr_ratio_aligned[i] < 0.7):  # strong volatility contraction
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above upper Donchian OR volatility contracts significantly
            if (close[i] > high_ma[i] or 
                atr_ratio_aligned[i] < 0.7):  # strong volatility contraction
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals