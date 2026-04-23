#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout + 1d ATR filter + volume confirmation + chop regime
- Donchian(20) breakout captures medium-term momentum
- 1d ATR filter: only trade when ATR(14) > 1.5 * ATR(50) to ensure sufficient volatility
- Volume confirmation (> 1.5x 20-period MA) reduces false breakouts
- Choppiness regime filter: CHOP(14) between 38.2 and 61.8 avoids extreme trending/choppy markets
- Designed for 4h timeframe to balance trade frequency and signal quality
- Works in bull via upside breakouts and in bear via downside breakouts
- Target: 20-50 trades/year per symbol (80-200 total over 4 years) to avoid fee drag
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d ATR for volatility filter
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
    atr_ratio = atr_14 / (atr_50 + 1e-10)
    
    # Align ATR ratio to 4h
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Calculate Donchian(20) on 4h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Choppiness Index: CHOP(14)
    # CHOP = 100 * log10(sum(TR over 14) / (max(high)-min(low) over 14)) / log10(14)
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(tr_14 / (max_high_14 - min_low_14 + 1e-10)) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50, 20, 14)  # Donchian, ATR50, volMA, chop
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(atr_ratio_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high AND sufficient volatility AND volume spike AND chop in range
            if (close[i] > donchian_high[i] and 
                atr_ratio_aligned[i] > 1.5 and 
                volume[i] > 1.5 * vol_ma[i] and
                38.2 <= chop[i] <= 61.8):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND sufficient volatility AND volume spike AND chop in range
            elif (close[i] < donchian_low[i] and 
                  atr_ratio_aligned[i] > 1.5 and 
                  volume[i] > 1.5 * vol_ma[i] and
                  38.2 <= chop[i] <= 61.8):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to opposite Donchian level OR volatility drops
            exit_signal = False
            if position == 1:
                # Exit long when price < Donchian low OR volatility insufficient
                if close[i] < donchian_low[i] or atr_ratio_aligned[i] < 1.2:
                    exit_signal = True
            elif position == -1:
                # Exit short when price > Donchian high OR volatility insufficient
                if close[i] > donchian_high[i] or atr_ratio_aligned[i] < 1.2:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian20_Breakout_1dATR_Volume_ChopFilter"
timeframe = "4h"
leverage = 1.0