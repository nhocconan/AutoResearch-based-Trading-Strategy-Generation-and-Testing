#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1dTrix_Volume_Crossover"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for TRIX calculation
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate TRIX(15): Triple EMA of 1-period log returns
    # TRIX = EMA(EMA(EMA(log(close)), 15), 15), 15) * 100
    log_close = np.log(close_1d)
    ema1 = pd.Series(log_close).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix = np.diff(ema3, prepend=ema3[0]) * 100  # Approximate derivative
    
    # Align TRIX to 6h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30
    
    for i in range(start_idx, n):
        if (np.isnan(trix_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        trix_val = trix_aligned[i]
        
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long: TRIX crosses above zero with volume
            if i > 0 and trix_aligned[i-1] <= 0 and trix_val > 0 and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero with volume
            elif i > 0 and trix_aligned[i-1] >= 0 and trix_val < 0 and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: TRIX crosses below zero
            if trix_val < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: TRIX crosses above zero
            if trix_val > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals