#!/usr/bin/env python3
# 4h_1d_camarilla_breakout_v1
# Strategy: Camarilla pivot levels from 1d with volume confirmation
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Camarilla levels act as strong support/resistance; price rejection at these levels with volume provides high-probability entries. Works in both bull and bear markets by capturing reversals at key levels.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_breakout_v1"
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # Camarilla formulas:
    # H4 = close + 1.1 * (high - low) / 2
    # H3 = close + 1.1 * (high - low) / 4
    # H2 = close + 1.1 * (high - low) / 6
    # H1 = close + 1.1 * (high - low) / 12
    # L1 = close - 1.1 * (high - low) / 12
    # L2 = close - 1.1 * (high - low) / 6
    # L3 = close - 1.1 * (high - low) / 4
    # L4 = close - 1.1 * (high - low) / 2
    
    prev_close = df_1d['close'].values
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    
    # Calculate Camarilla levels
    H4 = prev_close + 1.1 * (prev_high - prev_low) / 2
    H3 = prev_close + 1.1 * (prev_high - prev_low) / 4
    H2 = prev_close + 1.1 * (prev_high - prev_low) / 6
    H1 = prev_close + 1.1 * (prev_high - prev_low) / 12
    L1 = prev_close - 1.1 * (prev_high - prev_low) / 12
    L2 = prev_close - 1.1 * (prev_high - prev_low) / 6
    L3 = prev_close - 1.1 * (prev_high - prev_low) / 4
    L4 = prev_close - 1.1 * (prev_high - prev_low) / 2
    
    # Align Camarilla levels to 4h timeframe (1-day delay for previous day's levels)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    H2_aligned = align_htf_to_ltf(prices, df_1d, H2)
    H1_aligned = align_htf_to_ltf(prices, df_1d, H1)
    L1_aligned = align_htf_to_ltf(prices, df_1d, L1)
    L2_aligned = align_htf_to_ltf(prices, df_1d, L2)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    
    # Volume spike: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_spike = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(H4_aligned[i]) or np.isnan(L4_aligned[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Long setup: price breaks above H3 with volume spike (bullish breakout)
        # Short setup: price breaks below L3 with volume spike (bearish breakout)
        if (close[i] > H3_aligned[i] and volume_spike[i] and position != 1):
            position = 1
            signals[i] = 0.25
        elif (close[i] < L3_aligned[i] and volume_spike[i] and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: price returns to mean (H1/L1) or opposite extreme
        elif position == 1 and (close[i] < H1_aligned[i] or close[i] > H4_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > L1_aligned[i] or close[i] < L4_aligned[i]):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals