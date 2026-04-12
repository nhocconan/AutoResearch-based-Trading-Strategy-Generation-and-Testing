#!/usr/bin/env python3
"""
12h_1d_volume_momentum_reversal
Hypothesis: 12-hour strategy combining daily RSI extremes with volume spike confirmation.
Goes long when daily RSI < 30 with volume spike, short when RSI > 70 with volume spike.
Volume spike defined as current volume > 1.5x 20-period average.
Works in both bull and bear markets by fading extremes with volume confirmation.
Target: 20-30 trades/year (80-120 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for RSI and volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily RSI(14)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d = rsi_1d.values
    
    # Calculate 20-period average volume for spike detection
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align to 12h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if data not ready
        if (np.isnan(rsi_1d_aligned[i]) or np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike condition: current 12h volume > 1.5x daily average volume
        vol_spike = volume[i] > (vol_ma_1d_aligned[i] * 1.5)
        
        # Exit conditions: RSI returns to neutral zone
        if position == 1 and rsi_1d_aligned[i] > 50:
            position = 0
            signals[i] = 0.0
        elif position == -1 and rsi_1d_aligned[i] < 50:
            position = 0
            signals[i] = 0.0
        # Entry conditions
        elif vol_spike:
            if rsi_1d_aligned[i] < 30 and position != 1:
                position = 1
                signals[i] = 0.25
            elif rsi_1d_aligned[i] > 70 and position != -1:
                position = -1
                signals[i] = -0.25
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_volume_momentum_reversal"
timeframe = "12h"
leverage = 1.0