#!/usr/bin/env python3
"""
6h_1d_MACD_Signal_Crossover_Volume
Hypothesis: Use MACD signal line crossover on 1d timeframe for primary trend direction, combined with 6h price action confirmation and volume filter. MACD provides clear trend signals with reduced whipsaws, while volume confirmation ensures institutional participation. Works in bull markets by taking longs on MACD bullish crossovers above zero, and in bear markets by taking shorts on bearish crossovers below zero. Targets 15-25 trades/year by requiring alignment of MACD crossover, price confirmation in direction of trend, and volume > 1.3x 20-period average.
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
    
    # Get 1d data for MACD (HTF)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate MACD on 1d close
    # EMA12
    ema12 = np.full_like(close_1d, np.nan)
    k = 2 / (12 + 1)
    ema12[11] = np.mean(close_1d[:12])
    for i in range(12, len(close_1d)):
        ema12[i] = close_1d[i] * k + ema12[i-1] * (1 - k)
    
    # EMA26
    ema26 = np.full_like(close_1d, np.nan)
    k = 2 / (26 + 1)
    ema26[25] = np.mean(close_1d[:26])
    for i in range(26, len(close_1d)):
        ema26[i] = close_1d[i] * k + ema26[i-1] * (1 - k)
    
    # MACD line
    macd_line = ema12 - ema26
    
    # Signal line (EMA9 of MACD)
    signal_line = np.full_like(macd_line, np.nan)
    k = 2 / (9 + 1)
    # Find first valid MACD value
    first_valid = np.where(~np.isnan(macd_line))[0]
    if len(first_valid) > 0:
        start_idx = first_valid[0]
        signal_line[start_idx] = macd_line[start_idx]
        for i in range(start_idx + 1, len(macd_line)):
            signal_line[i] = macd_line[i] * k + signal_line[i-1] * (1 - k)
    
    # MACD histogram (for crossover detection)
    macd_hist = macd_line - signal_line
    
    # Align MACD components to 6h timeframe
    macd_line_aligned = align_htf_to_ltf(prices, df_1d, macd_line)
    signal_line_aligned = align_htf_to_ltf(prices, df_1d, signal_line)
    macd_hist_aligned = align_htf_to_ltf(prices, df_1d, macd_hist)
    
    # Volume confirmation: current volume > 1.3 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_confirm = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(26, 20)  # need MACD warmup and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(macd_line_aligned[i]) or np.isnan(signal_line_aligned[i]) or 
            np.isnan(macd_hist_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: MACD bullish crossover (hist crosses above zero) with volume confirmation
            if (macd_hist_aligned[i] > 0 and macd_hist_aligned[i-1] <= 0 and vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: MACD bearish crossover (hist crosses below zero) with volume confirmation
            elif (macd_hist_aligned[i] < 0 and macd_hist_aligned[i-1] >= 0 and vol_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: MACD bearish crossover or loss of momentum
            if macd_hist_aligned[i] < 0 and macd_hist_aligned[i-1] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: MACD bullish crossover or loss of momentum
            if macd_hist_aligned[i] > 0 and macd_hist_aligned[i-1] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1d_MACD_Signal_Crossover_Volume"
timeframe = "6h"
leverage = 1.0