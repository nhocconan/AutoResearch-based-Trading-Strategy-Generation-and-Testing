#!/usr/bin/env python3
"""
Hypothesis: 4h TRIX momentum with volume spike confirmation and 12h EMA50 trend filter.
Long when TRIX crosses above zero AND volume > 2.0x 20-period average AND close > 12h EMA50.
Short when TRIX crosses below zero AND volume > 2.0x 20-period average AND close < 12h EMA50.
Exit when TRIX crosses back through zero in opposite direction.
Uses discrete position sizing (0.25) to minimize fee churn. Targets 19-50 trades/year per symbol.
TRIX (triple exponential average) filters noise and captures sustained momentum. Volume confirmation ensures institutional participation.
12h EMA50 provides medium-term trend alignment to avoid counter-trend whipsaws. Works in both bull and bear markets by
only taking trades in the direction of the 12h trend, reducing false breakouts during ranging periods.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data for EMA50 trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 1:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50 for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate TRIX (15,9,9) on primary timeframe
    # TRIX = EMA(EMA(EMA(close, 15), 15), 15) - 1 period ago
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix = (ema3 - np.roll(ema3, 1)) / np.roll(ema3, 1) * 100
    trix[0] = 0  # First value has no previous
    
    # Align HTF indicators to 4h timeframe
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 50, 15*3)  # Ensure warmup for EMA50 and TRIX
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(trix[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        trix_val = trix[i]
        trix_prev = trix[i-1]
        
        if position == 0:
            # Long: TRIX crosses above zero AND volume spike AND close > 12h EMA50
            if (trix_val > 0 and trix_prev <= 0 and 
                volume[i] > 2.0 * vol_ma_val and 
                close[i] > ema50_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero AND volume spike AND close < 12h EMA50
            elif (trix_val < 0 and trix_prev >= 0 and 
                  volume[i] > 2.0 * vol_ma_val and 
                  close[i] < ema50_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            # Primary exit: TRIX crosses back through zero in opposite direction
            if position == 1 and trix_val < 0 and trix_prev >= 0:
                exit_signal = True
            elif position == -1 and trix_val > 0 and trix_prev <= 0:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_TRIX_12hEMA50_VolumeSpike"
timeframe = "4h"
leverage = 1.0