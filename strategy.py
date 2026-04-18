#!/usr/bin/env python3
"""
12h_Donchian_Breakout_Volume_Trend
Hypothesis: On 12h timeframe, use Donchian(20) breakout for trend entries, confirmed by 1D volume spike (>1.5x 20-period average) and 1D EMA34 trend filter. Long when price breaks above upper band with volume confirmation and close > EMA34; short when breaks below lower band with volume confirmation and close < EMA34. Exit on opposite break. Targets 15-25 trades/year with position size 0.25, designed to capture major trends while avoiding whipsaws in ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(20, n):
        upper[i] = np.max(high[i-20:i])
        lower[i] = np.min(low[i-20:i])
    
    # Get 1D data for volume and EMA filters
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate volume average (20-period) on 1D
    vol_ma = np.full(len(volume_1d), np.nan)
    for i in range(20, len(volume_1d)):
        vol_ma[i] = np.mean(volume_1d[i-20:i])
    
    # Volume spike condition: current volume > 1.5x 20-period average
    vol_spike = np.zeros(len(volume_1d), dtype=bool)
    for i in range(20, len(volume_1d)):
        if not np.isnan(vol_ma[i]) and vol_ma[i] > 0:
            vol_spike[i] = volume_1d[i] > (1.5 * vol_ma[i])
    
    # Calculate EMA34 on 1D close
    ema34 = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 34:
        alpha = 2 / (34 + 1)
        ema34[33] = np.mean(close_1d[0:34])
        for i in range(34, len(close_1d)):
            ema34[i] = alpha * close_1d[i] + (1 - alpha) * ema34[i-1]
    
    # Align 1D indicators to 12h timeframe
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike.astype(float))
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need Donchian
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(vol_spike_aligned[i]) or np.isnan(ema34_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price breaks above upper band with volume spike and close > EMA34
            if (close[i] > upper[i] and vol_spike_aligned[i] > 0.5 and close[i] > ema34_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below lower band with volume spike and close < EMA34
            elif (close[i] < lower[i] and vol_spike_aligned[i] > 0.5 and close[i] < ema34_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price breaks below lower band
            if close[i] < lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above upper band
            if close[i] > upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian_Breakout_Volume_Trend"
timeframe = "12h"
leverage = 1.0