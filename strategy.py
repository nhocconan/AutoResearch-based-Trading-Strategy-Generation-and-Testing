#!/usr/bin/env python3
"""
4h_1d_TRIX_ZeroCross_Volume_Confirm
Hypothesis: Use 1-hour TRIX zero-cross as momentum signal filtered by 1-day trend (price vs 50 EMA) and volume confirmation. TRIX filters noise and captures momentum shifts. Long when TRIX crosses above zero in uptrend (price > 50 EMA), short when crosses below zero in downtrend (price < 50 EMA). Volume > 1.3x 20-period average confirms breakout strength. Targets 20-35 trades/year by requiring TRIX cross + trend alignment + volume spike. Works in bull markets via momentum continuation and in bear via counter-trend momentum exhaustion signals.
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
    
    # Get 1h data for TRIX calculation
    df_1h = get_htf_data(prices, '1h')
    close_1h = df_1h['close'].values
    
    # Calculate TRIX: triple EMA of log returns, then ROC
    # EMA1
    ema1 = np.full_like(close_1h, np.nan)
    ema1[0] = close_1h[0]
    for i in range(1, len(close_1h)):
        ema1[i] = 0.15 * close_1h[i] + 0.85 * ema1[i-1]  # alpha = 2/(12+1) for 12-period EMA
    
    # EMA2 of EMA1
    ema2 = np.full_like(close_1h, np.nan)
    ema2[0] = ema1[0]
    for i in range(1, len(close_1h)):
        ema2[i] = 0.15 * ema1[i] + 0.85 * ema2[i-1]
    
    # EMA3 of EMA2
    ema3 = np.full_like(close_1h, np.nan)
    ema3[0] = ema2[0]
    for i in range(1, len(close_1h)):
        ema3[i] = 0.15 * ema2[i] + 0.85 * ema3[i-1]
    
    # TRIX = 100 * (EMA3[t] - EMA3[t-1]) / EMA3[t-1]
    trix = np.full_like(close_1h, np.nan)
    for i in range(1, len(close_1h)):
        if ema3[i-1] != 0:
            trix[i] = 100 * (ema3[i] - ema3[i-1]) / ema3[i-1]
    
    # Align TRIX to 4h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_1h, trix)
    
    # Get 1d data for 50 EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 50-period EMA on daily close
    ema50_1d = np.full_like(close_1d, np.nan)
    for i in range(len(close_1d)):
        if i < 50:
            ema50_1d[i] = np.mean(close_1d[:i+1]) if i > 0 else close_1d[0]
        else:
            ema50_1d[i] = 0.0392 * close_1d[i] + 0.9608 * ema50_1d[i-1]  # alpha = 2/(50+1)
    
    # Align 50 EMA to 4h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: current volume > 1.3 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_confirm = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need TRIX and EMA50 warmed up
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(trix_aligned[i]) or np.isnan(trix_aligned[i-1]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: TRIX crosses above zero, price above 50 EMA (uptrend), volume confirmation
            if (trix_aligned[i-1] <= 0 and trix_aligned[i] > 0 and 
                close[i] > ema50_1d_aligned[i] and vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: TRIX crosses below zero, price below 50 EMA (downtrend), volume confirmation
            elif (trix_aligned[i-1] >= 0 and trix_aligned[i] < 0 and 
                  close[i] < ema50_1d_aligned[i] and vol_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: TRIX crosses below zero (momentum loss) or price breaks below 50 EMA (trend change)
            if (trix_aligned[i] < 0 or close[i] < ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: TRIX crosses above zero (momentum loss) or price breaks above 50 EMA (trend change)
            if (trix_aligned[i] > 0 or close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1d_TRIX_ZeroCross_Volume_Confirm"
timeframe = "4h"
leverage = 1.0