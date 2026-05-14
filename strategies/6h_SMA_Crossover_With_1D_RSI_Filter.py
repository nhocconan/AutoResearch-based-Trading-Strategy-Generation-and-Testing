#!/usr/bin/env python3
"""
6h_SMA_Crossover_With_1D_RSI_Filter
Hypothesis: On 6h timeframe, use SMA(20) crossing above/below SMA(50) for trend signals, filtered by 1D RSI(14) to avoid counter-trend trades. Long when SMA20>SMA50 and RSI<60 (not overbought); short when SMA20<SMA50 and RSI>40 (not oversold). Exit on opposite cross. This captures trends while avoiding extremes, reducing whipsaws in 2022 crash and avoiding overextended entries in ranging markets. Targets 15-25 trades/year by requiring SMA cross + RSI filter, with position size 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Calculate SMA20 and SMA50 on 6h data
    sma20 = np.full(n, np.nan)
    sma50 = np.full(n, np.nan)
    
    for i in range(20, n):
        sma20[i] = np.mean(close[i-20:i])
    for i in range(50, n):
        sma50[i] = np.mean(close[i-50:i])
    
    # Get 1D data for RSI filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate RSI(14) on 1D
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(close_1d, np.nan)
    avg_loss = np.full_like(close_1d, np.nan)
    
    for i in range(14, len(close_1d)):
        if i == 14:
            avg_gain[i] = np.mean(gain[0:14])
            avg_loss[i] = np.mean(loss[0:14])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Align RSI to 6h timeframe (wait for bar close)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need SMA50
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(sma20[i]) or np.isnan(sma50[i]) or 
            np.isnan(rsi_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: SMA20 > SMA50 and RSI not overbought (<60)
            if (sma20[i] > sma50[i] and rsi_1d_aligned[i] < 60):
                signals[i] = 0.25
                position = 1
            # Short entry: SMA20 < SMA50 and RSI not oversold (>40)
            elif (sma20[i] < sma50[i] and rsi_1d_aligned[i] > 40):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: SMA20 crosses below SMA50
            if sma20[i] < sma50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: SMA20 crosses above SMA50
            if sma20[i] > sma50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_SMA_Crossover_With_1D_RSI_Filter"
timeframe = "6h"
leverage = 1.0