#!/usr/bin/env python3
"""
4h_Donchian20_Volume_HMA_Trend
Hypothesis: Use 4h Donchian(20) breakout with HMA(21) trend filter and volume confirmation.
Long when price breaks above upper band with rising HMA and volume spike; short when breaks below lower band with falling HMA and volume spike.
Works in bull markets by catching breakouts and in bear markets by catching breakdowns. Volume filter reduces false breakouts.
Target: 20-40 trades/year on 4h timeframe for low fee drag and high signal quality.
"""

import numpy as np
import pandas as pd
from mats import HMA  # HMA is available via mats module

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data for HMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate HMA(21) on daily close
    hma_21_1d = HMA(df_1d['close'].values, 21)
    hma_21_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_21_1d)
    
    # Calculate Donchian bands (20-period) on 4h data
    high_20 = prices['high'].rolling(window=20, min_periods=20).max().values
    low_20 = prices['low'].rolling(window=20, min_periods=20).min().values
    
    # Volume spike: current volume > 1.8 * 20-period average
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    volume_spike = prices['volume'].values > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if indicators not ready
        if (np.isnan(hma_21_1d_aligned[i]) or 
            np.isnan(high_20[i]) or np.isnan(low_20[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian band with rising HMA and volume spike
            if (price > high_20[i] and 
                hma_21_1d_aligned[i] > hma_21_1d_aligned[i-1] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian band with falling HMA and volume spike
            elif (price < low_20[i] and 
                  hma_21_1d_aligned[i] < hma_21_1d_aligned[i-1] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below HMA or breaks below lower Donchian band
            if (price < hma_21_1d_aligned[i] or 
                price < low_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above HMA or breaks above upper Donchian band
            if (price > hma_21_1d_aligned[i] or 
                price > high_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Volume_HMA_Trend"
timeframe = "4h"
leverage = 1.0