#!/usr/bin/env python3
# Hypothesis: 6h Donchian(20) breakout with 1d ATR regime filter and volume confirmation.
# Uses Donchian channels for breakout detection, 1d ATR ratio (ATR10/ATR30) to filter low-volatility chop,
# and volume > 1.5x 20-bar average for confirmation. Designed for low trade frequency (<100 total 6h trades)
# to minimize fee drag while capturing strong momentum moves in both bull and bear markets.
# Works in bull markets via breakout continuation and in bear markets via volatility expansion bursts.

name = "6h_Donchian20_1dATRRegime_VolumeConfirm_v1"
timeframe = "6h"
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
    
    # Calculate 1d ATR10 and ATR30 for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # True Range calculation
    tr1 = pd.Series(df_1d['high']).values - pd.Series(df_1d['low']).values
    tr2 = np.abs(pd.Series(df_1d['high']).values - pd.Series(df_1d['close']).shift(1).values)
    tr3 = np.abs(pd.Series(df_1d['low']).values - pd.Series(df_1d['close']).shift(1).values)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_10 = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    atr_30 = pd.Series(tr).rolling(window=30, min_periods=30).mean().values
    atr_ratio = atr_10 / atr_30  # >1 indicates expanding volatility
    
    # Align ATR ratio to 6h timeframe
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Calculate 6h Donchian channels (20-period)
    lookback_dc = 20
    dc_high = pd.Series(high).rolling(window=lookback_dc, min_periods=lookback_dc).max().shift(1).values
    dc_low = pd.Series(low).rolling(window=lookback_dc, min_periods=lookback_dc).min().shift(1).values
    
    # Calculate average volume for confirmation (20-period)
    lookback_vol = 20
    avg_volume = pd.Series(volume).rolling(window=lookback_vol, min_periods=lookback_vol).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(max(lookback_dc, lookback_vol, 30), n):
        # Skip if any required data is NaN
        if (np.isnan(dc_high[i]) or 
            np.isnan(dc_low[i]) or 
            np.isnan(atr_ratio_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade when volatility is expanding (ATR ratio > 1.1)
        if atr_ratio_aligned[i] <= 1.1:
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian high, volume spike
            if (high[i] > dc_high[i] and 
                volume[i] > 1.5 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian low, volume spike
            elif (low[i] < dc_low[i] and 
                  volume[i] > 1.5 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Donchian low or volume drops significantly
            if (low[i] < dc_low[i]) or (volume[i] < 0.5 * avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # Maintain position
        elif position == -1:
            # EXIT SHORT: Price breaks above Donchian high or volume drops significantly
            if (high[i] > dc_high[i]) or (volume[i] < 0.5 * avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # Maintain position
    
    return signals