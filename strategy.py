#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_ewma_crossover_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Calculate EWMA on daily closes
    close_1d = df_1d['close'].values
    ewma_fast = pd.Series(close_1d).ewm(span=9, adjust=False, min_periods=9).mean().values
    ewma_slow = pd.Series(close_1d).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align EWMA to 6h timeframe
    ewma_fast_aligned = align_htf_to_ltf(prices, df_1d, ewma_fast)
    ewma_slow_aligned = align_htf_to_ltf(prices, df_1d, ewma_slow)
    
    # Volume filter: 6h volume > 1.5x 50-period average
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(ewma_fast_aligned[i]) or np.isnan(ewma_slow_aligned[i]) or
            np.isnan(vol_ma_50[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        
        # Volume confirmation
        vol_confirm = volume_current > 1.5 * vol_ma_50[i]
        
        # EWMA crossover signals
        cross_up = ewma_fast_aligned[i] > ewma_slow_aligned[i]
        cross_down = ewma_fast_aligned[i] < ewma_slow_aligned[i]
        
        # Entry conditions with volume confirmation
        enter_long = False
        enter_short = False
        
        # Long: Fast EWMA crosses above slow EWMA with volume
        if i > 50 and cross_up and not (ewma_fast_aligned[i-1] > ewma_slow_aligned[i-1]) and vol_confirm:
            enter_long = True
        
        # Short: Fast EWMA crosses below slow EWMA with volume
        if i > 50 and cross_down and not (ewma_fast_aligned[i-1] < ewma_slow_aligned[i-1]) and vol_confirm:
            enter_short = True
        
        # Exit conditions: opposite crossover
        exit_long = i > 50 and cross_down and not (ewma_fast_aligned[i-1] > ewma_slow_aligned[i-1])
        exit_short = i > 50 and cross_up and not (ewma_fast_aligned[i-1] < ewma_slow_aligned[i-1])
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: 6s EWMA crossover on daily timeframe with volume confirmation.
# Uses fast (9) and slow (21) EWMA on daily closes aligned to 6h timeframe.
# Enters long when fast EWMA crosses above slow EWMA with volume > 1.5x 50-period average.
# Enters short when fast EWMA crosses below slow EWMA with volume > 1.5x 50-period average.
# Exits on opposite crossover.
# Volume filter reduces false signals and controls trade frequency.
# Position size 0.25 manages risk. Target: 20-40 trades per year (80-160 total over 4 years).
# Works in both bull and bear markets by capturing trend changes with volume confirmation.
# 6h timeframe provides balance between signal quality and trade frequency.
# EWMA crossover is less prone to whipsaw than SMA crossover in volatile markets.