#!/usr/bin/env python3
"""
1d_1w_Volume_Weighted_MA_Crossover_Trend
Hypothesis: Use volume-weighted moving average crossover on 1d with 1w trend filter to capture medium-term trends while avoiding whipsaws. The VWMA reacts faster to price moves on high volume, making it effective in both bull and bear markets. The 1w EMA filter ensures we only trade in the direction of the higher timeframe trend. Targets 10-25 trades/year by requiring VWMA crossover, volume confirmation, and 1w trend alignment. Works in bull markets by taking longs when price is above 1w EMA and VWMA crosses up, and in bear markets by taking shorts when price is below 1w EMA and VWMA crosses down.
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
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA(21) for trend filter
    ema_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 21:
        ema_1w[20] = close_1w[:21].mean()
        for i in range(21, len(close_1w)):
            ema_1w[i] = (close_1w[i] * 2 / (21 + 1)) + (ema_1w[i-1] * (21 - 1) / (21 + 1))
    
    # Align 1w EMA to 1d timeframe
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate VWMA(20) on 1d
    vwma = np.full(n, np.nan)
    for i in range(20, n):
        vwma[i] = np.dot(close[i-20:i+1], volume[i-20:i+1]) / volume[i-20:i+1].sum()
    
    # Calculate VWMA(50) on 1d
    vwma_long = np.full(n, np.nan)
    for i in range(50, n):
        vwma_long[i] = np.dot(close[i-50:i+1], volume[i-50:i+1]) / volume[i-50:i+1].sum()
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need VWMA(50) and EMA seeded
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(vwma[i]) or np.isnan(vwma_long[i]) or 
            np.isnan(ema_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price above 1w EMA, VWMA(20) crosses above VWMA(50)
            if (close[i] > ema_1w_aligned[i] and 
                vwma[i] > vwma_long[i] and 
                vwma[i-1] <= vwma_long[i-1]):
                signals[i] = 0.25
                position = 1
            # Short entry: price below 1w EMA, VWMA(20) crosses below VWMA(50)
            elif (close[i] < ema_1w_aligned[i] and 
                  vwma[i] < vwma_long[i] and 
                  vwma[i-1] >= vwma_long[i-1]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: VWMA(20) crosses below VWMA(50) or price crosses below 1w EMA
            if (vwma[i] < vwma_long[i] or 
                close[i] < ema_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: VWMA(20) crosses above VWMA(50) or price crosses above 1w EMA
            if (vwma[i] > vwma_long[i] or 
                close[i] > ema_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1w_Volume_Weighted_MA_Crossover_Trend"
timeframe = "1d"
leverage = 1.0