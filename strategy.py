#!/usr/bin/env python3
"""
4h_12h_1d_SMA_Crossover_Volume_Strategy
Hypothesis: Use 12h SMA(34) trend as primary filter, with 4h SMA(10) crossover for entry timing, and volume confirmation (>1.8x 20-period average). This avoids whipsaws by requiring alignment between 12h trend and 4h momentum. Works in bull markets via long entries when 12h trend up and 4h SMA crosses above; in bear markets via short entries when 12h trend down and 4h SMA crosses below. Targets 20-30 trades/year with strict conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for SMA trend (HTF)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 12h SMA(34)
    sma_12h = np.full_like(close_12h, np.nan)
    for i in range(33, len(close_12h)):
        sma_12h[i] = np.mean(close_12h[i-33:i+1])
    
    # Align 12h SMA to 4h timeframe
    sma_12h_aligned = align_htf_to_ltf(prices, df_12h, sma_12h)
    
    # Calculate 4h SMA(10) and SMA(30) for crossover
    sma_10 = np.full(n, np.nan)
    sma_30 = np.full(n, np.nan)
    for i in range(9, n):
        sma_10[i] = np.mean(close[i-9:i+1])
    for i in range(29, n):
        sma_30[i] = np.mean(close[i-29:i+1])
    
    # Volume confirmation: current volume > 1.8 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_confirm = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # need SMA30
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(sma_12h_aligned[i]) or np.isnan(sma_10[i]) or 
            np.isnan(sma_30[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: 12h uptrend (close > SMA), 4h SMA10 crosses above SMA30, with volume
            if (close[i] > sma_12h_aligned[i] and 
                sma_10[i] > sma_30[i] and 
                sma_10[i-1] <= sma_30[i-1] and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: 12h downtrend (close < SMA), 4h SMA10 crosses below SMA30, with volume
            elif (close[i] < sma_12h_aligned[i] and 
                  sma_10[i] < sma_30[i] and 
                  sma_10[i-1] >= sma_30[i-1] and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: 12h trend turns down OR 4h SMA10 crosses below SMA30
            if (close[i] < sma_12h_aligned[i] or 
                (sma_10[i] < sma_30[i] and sma_10[i-1] >= sma_30[i-1])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: 12h trend turns up OR 4h SMA10 crosses above SMA30
            if (close[i] > sma_12h_aligned[i] or 
                (sma_10[i] > sma_30[i] and sma_10[i-1] <= sma_30[i-1])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_12h_1d_SMA_Crossover_Volume_Strategy"
timeframe = "4h"
leverage = 1.0