#!/usr/bin/env python3
# 1d_Weekly_RSI_Reversal_With_Volume_Confirmation
# Hypothesis: Uses weekly RSI(14) to detect overbought/oversold conditions on the weekly chart,
# combined with daily volume spikes to confirm reversals. Enters long when weekly RSI < 30 and
# daily volume > 1.5x 20-day average volume. Enters short when weekly RSI > 70 and volume spike.
# Exits when RSI returns to neutral zone (40-60). Designed for low-frequency, high-conviction
# trades to avoid overtrading and work in both bull and bear markets by catching extremes.

name = "1d_Weekly_RSI_Reversal_With_Volume_Confirmation"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 20-day average volume for volume spike filter
    vol_avg_20 = np.zeros(n)
    vol_sum = 0.0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i - 20]
        if i >= 19:  # Start from index 19 (20th element)
            vol_avg_20[i] = vol_sum / 20.0
    
    # Get weekly data for RSI calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly RSI(14)
    delta = np.diff(close_1w)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close_1w)
    avg_loss = np.zeros_like(close_1w)
    
    # Initial average gain/loss
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    # Wilder's smoothing
    for i in range(14, len(close_1w)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.zeros_like(close_1w)
    rsi_1w = np.zeros_like(close_1w)
    for i in range(14, len(close_1w)):
        if avg_loss[i] != 0:
            rs[i] = avg_gain[i] / avg_loss[i]
            rsi_1w[i] = 100 - (100 / (1 + rs[i]))
        else:
            rsi_1w[i] = 100.0  # Avoid division by zero
    
    # Align weekly RSI to daily timeframe
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure sufficient warmup for volume average
    
    for i in range(start_idx, n):
        if np.isnan(rsi_1w_aligned[i]) or np.isnan(vol_avg_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike condition: current volume > 1.5x 20-day average
        volume_spike = volume[i] > 1.5 * vol_avg_20[i]
        
        if position == 0:
            # Long: Oversold weekly RSI + volume spike
            if rsi_1w_aligned[i] < 30 and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: Overbought weekly RSI + volume spike
            elif rsi_1w_aligned[i] > 70 and volume_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: RSI returns to neutral zone (40-60)
            if rsi_1w_aligned[i] >= 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: RSI returns to neutral zone (40-60)
            if rsi_1w_aligned[i] <= 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals