#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout + 1d ATR regime filter + volume spike confirmation.
- Primary timeframe: 12h for execution, HTF: 1d for ATR regime and volume confirmation.
- Entry: Long when price breaks above 20-period Donchian high with volume > 1.8x 20-period volume MA AND 1d ATR(14) > 1.5x 50-period ATR MA (high volatility regime).
         Short when price breaks below 20-period Donchian low with volume > 1.8x 20-period volume MA AND 1d ATR(14) > 1.5x 50-period ATR MA.
- Direction filter: ATR regime ensures we only trade in high volatility environments where breakouts are meaningful.
- Volume confirmation reduces false breakouts.
- Exit: Opposite Donchian breakout (long exits at Donchian low, short exits at Donchian high) or ATR regime collapse (ATR < 1.2x 50-period ATR MA).
- Discrete signal size: 0.25 to balance return and drawdown control.
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
- Works in bull via buying breakouts in uptrends, in bear via selling breakdowns in downtrends, with volatility filter avoiding choppy markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d ATR(14) for regime filter
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
    tr1[0] = high_1d[0] - low_1d[0]  # First period
    tr2[0] = high_1d[0] - close_1d[0]  # First period
    tr3[0] = low_1d[0] - close_1d[0]   # First period
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 50-period ATR MA for regime threshold
    atr_ma_50 = pd.Series(atr_14).rolling(window=50, min_periods=50).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    atr_ma_50_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_50)
    
    # Volume confirmation: current volume > 1.8 * 20-period volume MA (using 1d volume)
    volume_1d = df_1d['volume'].values
    volume_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (1.8 * volume_ma_20)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    
    # Calculate 12h Donchian channels (20-period)
    if len(high) < 20:
        return np.zeros(n)
    
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need ATR(14) with 50-period MA and Donchian(20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(atr_1d_aligned[i]) or np.isnan(atr_ma_50_aligned[i]) or 
            np.isnan(volume_spike_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # ATR regime filter: only trade when current ATR > 1.5 * 50-period ATR MA (high volatility)
        high_volatility_regime = atr_1d_aligned[i] > (1.5 * atr_ma_50_aligned[i])
        # ATR regime collapse filter: exit when ATR < 1.2 * 50-period ATR MA
        low_volatility_exit = atr_1d_aligned[i] < (1.2 * atr_ma_50_aligned[i])
        
        if position == 0:
            # Long: price breaks above Donchian high with volume spike AND high volatility regime
            if (close[i] > highest_high[i] and volume_spike_aligned[i] and 
                high_volatility_regime):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with volume spike AND high volatility regime
            elif (close[i] < lowest_low[i] and volume_spike_aligned[i] and 
                  high_volatility_regime):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price returns below Donchian low OR volatility regime collapses
            if (close[i] < lowest_low[i] or low_volatility_exit):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns above Donchian high OR volatility regime collapses
            if (close[i] > highest_high[i] or low_volatility_exit):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dATRRegime_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0