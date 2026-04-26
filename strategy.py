#!/usr/bin/env python3
"""
12h_Donchian20_Breakout_VolumeSpike_ChopFilter
Hypothesis: 12h Donchian channel breakout with volume spike and chop regime filter.
Long when price breaks above 20-bar high with volume spike in trending market (CHOP < 38.2).
Short when price breaks below 20-bar low with volume spike in trending market.
Uses ATR-based stoploss via signal=0 when price closes outside channel.
Designed for 12h timeframe to target 12-37 trades/year, works in bull/bear via trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for chop filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate Chopiness Index on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    
    # ATR(14)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Max/min high/low over 14 periods
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chopiness Index: 100 * log10(sum(TR14) / (max_high_14 - min_low_14)) / log10(14)
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    denominator = max_high_14 - min_low_14
    chop = np.where(denominator > 0, 100 * np.log10(sum_tr_14 / denominator) / np.log10(14), 100)
    
    # Align chop to 12h
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Donchian Channel (20-period) on 12h
    period = 20
    donchian_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    donchian_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # ATR(10) for stoploss
    atr_period = 10
    tr_12h = np.maximum(np.abs(high[1:] - low[:-1]), np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1]))
    tr_12h = np.concatenate([[np.nan], tr_12h])
    atr_10 = pd.Series(tr_12h).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Volume confirmation: volume > 2.0x 20-period MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(period, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(volume_spike[i]) or
            np.isnan(atr_10[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high with volume spike in trending market (CHOP < 38.2)
            if (close[i] > donchian_high[i] and 
                volume_spike[i] and chop_aligned[i] < 38.2):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with volume spike in trending market
            elif (close[i] < donchian_low[i] and 
                  volume_spike[i] and chop_aligned[i] < 38.2):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price closes below Donchian low OR chop becomes too high (choppy market)
            if (close[i] < donchian_low[i] or chop_aligned[i] > 61.8):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price closes above Donchian high OR chop becomes too high
            if (close[i] > donchian_high[i] or chop_aligned[i] > 61.8):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Donchian20_Breakout_VolumeSpike_ChopFilter"
timeframe = "12h"
leverage = 1.0