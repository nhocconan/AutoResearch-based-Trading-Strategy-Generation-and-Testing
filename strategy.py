#!/usr/bin/env python3
# Hypothesis: 12h Donchian(20) breakout with 1d ATR regime filter and volume confirmation.
# Long when price breaks above 20-period Donchian high and ATR(14) < ATR(50) (low volatility regime) with volume > 1.5x 20-bar average.
# Short when price breaks below 20-period Donchian low and ATR(14) < ATR(50) with volume > 1.5x 20-bar average.
# Uses discrete sizing 0.25 to target 12-37 trades/year on 12h timeframe.
# Donchian channels provide structural breakouts; ATR filter avoids high-volatility chop; volume confirms momentum.
# Designed for fewer, higher-quality trades to minimize fee drag while maintaining edge in both bull and bear markets.

name = "12h_Donchian20_1dATRRegime_VolumeConfirm"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d ATR(14) and ATR(50) for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align with index
    
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_50 = pd.Series(tr).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # ATR regime: low volatility when ATR(14) < ATR(50)
    atr_regime = atr_14 < atr_50
    atr_regime_aligned = align_htf_to_ltf(prices, df_1d, atr_regime.astype(float))
    
    # Calculate Donchian channels (20-period) from 1d data
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Calculate average volume for confirmation (20-period)
    lookback = 20
    avg_volume = pd.Series(volume).rolling(window=lookback, min_periods=lookback).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback, n):  # Start after sufficient data
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(atr_regime_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian high, low volatility regime, volume spike
            if (high[i] > donchian_high_aligned[i] and 
                atr_regime_aligned[i] > 0.5 and 
                volume[i] > 1.5 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian low, low volatility regime, volume spike
            elif (low[i] < donchian_low_aligned[i] and 
                  atr_regime_aligned[i] > 0.5 and 
                  volume[i] > 1.5 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Donchian low OR volatility increases (ATR regime fails)
            if (low[i] < donchian_low_aligned[i] or 
                atr_regime_aligned[i] <= 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Donchian high OR volatility increases (ATR regime fails)
            if (high[i] > donchian_high_aligned[i] or 
                atr_regime_aligned[i] <= 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals