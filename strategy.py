#!/usr/bin/env python3
"""
4h_1d_Donchian_Breakout_Volume_SMA200_Filter_v1
Hypothesis: 4h timeframe with Donchian channel breakout (20-period), volume confirmation (>1.5x average), and SMA200 trend filter.
Breakouts above upper band go long only when price > SMA200; breakdowns below lower band go short only when price < SMA200.
Designed for ~20-50 trades/year on 4h by requiring trend alignment and volume confirmation. Works in bull/bear markets by only taking trend-aligned breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Donchian_Breakout_Volume_SMA200_Filter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for SMA200
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d SMA200 for trend filter
    close_1d = df_1d['close'].values
    sma_200 = pd.Series(close_1d).rolling(window=200, min_periods=200).mean().values
    sma_200_aligned = align_htf_to_ltf(prices, df_1d, sma_200)
    
    # Calculate 4h Donchian channel (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average (20 period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(sma_200_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume spike: current volume > 1.5x average
        volume_spike = volume[i] > vol_ma[i] * 1.5
        
        # Trend filter: price above/below 1d SMA200
        above_sma200 = close[i] > sma_200_aligned[i]
        below_sma200 = close[i] < sma_200_aligned[i]
        
        # Entry conditions: Donchian breakout with volume and trend alignment
        long_entry = (close[i] > high_max[i]) and volume_spike and above_sma200
        short_entry = (close[i] < low_min[i]) and volume_spike and below_sma200
        
        # Exit conditions: return to opposite Donchian band or trend reversal
        long_exit = (close[i] < low_min[i]) or (close[i] < sma_200_aligned[i])
        short_exit = (close[i] > high_max[i]) or (close[i] > sma_200_aligned[i])
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals