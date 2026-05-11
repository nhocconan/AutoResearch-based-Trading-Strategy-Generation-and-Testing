#!/usr/bin/env python3
"""
1d_PhaseShift_Signal_With_Volume_Confirmation
Hypothesis: Uses 1-day price momentum with phase-shift oscillator (using Hilbert transform concept via SMA crossover timing) 
combined with weekly trend filter and volume confirmation. Designed for low trade frequency (<25/year) to avoid fee drag
while capturing directional moves in both bull and bear markets. The phase-shift signal helps identify momentum shifts
before they become extreme, reducing whipsaw.
"""

name = "1d_PhaseShift_Signal_With_Volume_Confirmation"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1d OHLCV
    close = prices['close'].values
    volume = prices['volume'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # --- Weekly EMA50 for trend filter ---
    close_1w = df_1w['close']
    ema_50_1w = close_1w.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # --- Phase-shift signal: Fast SMA (5) vs Slow SMA (20) crossover with delay ---
    # This creates a momentum signal that anticipates turns
    sma_fast = pd.Series(close).rolling(window=5, min_periods=5).mean().values
    sma_slow = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    
    # Phase signal: 1 when fast > slow, -1 when fast < smooth
    phase_signal = np.where(sma_fast > sma_slow, 1, -1)
    
    # --- Volume confirmation: 1.5x 20-period volume SMA ---
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_sma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(sma_fast[i]) or
            np.isnan(sma_slow[i]) or
            np.isnan(vol_sma[i])):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Determine weekly trend
        weekly_uptrend = close[i] > ema_50_1w_aligned[i]
        weekly_downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Entry conditions: phase shift aligned with weekly trend + volume confirmation
        long_entry = (phase_signal[i] == 1) and weekly_uptrend and volume_confirm[i]
        short_entry = (phase_signal[i] == -1) and weekly_downtrend and volume_confirm[i]
        
        # Exit conditions: phase reversal or loss of weekly trend
        long_exit = (phase_signal[i] == -1) or (close[i] < ema_50_1w_aligned[i])
        short_exit = (phase_signal[i] == 1) or (close[i] > ema_50_1w_aligned[i])
        
        if position == 0:
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                if long_exit:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                if short_exit:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals