#!/usr/bin/env python3
name = "4H_Daily_Trix_Trend_Reversal"
timeframe = "4h"
leverage = 1.0

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
    
    # Get daily data for TRIX and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate TRIX (Triple Exponential Average) - 15-period
    # TRIX = EMA(EMA(EMA(close, 15), 15), 15)
    ema1 = pd.Series(close_1d).ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    trix_raw = ema3.pct_change() * 100  # Percentage change
    trix = trix_raw.fillna(0).values
    
    # TRIX signal line (9-period EMA of TRIX)
    trix_signal = pd.Series(trix).ewm(span=9, adjust=False, min_periods=9).mean().values
    
    # Align TRIX and signal line to 4h
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix)
    trix_signal_aligned = align_htf_to_ltf(prices, df_1d, trix_signal)
    
    # Volume confirmation: current volume > 1.3x 20-period average
    volume_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_avg * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(trix_aligned[i]) or np.isnan(trix_signal_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: TRIX crosses above signal line + volume confirmation
            if trix_aligned[i] > trix_signal_aligned[i] and trix_aligned[i-1] <= trix_signal_aligned[i-1] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: TRIX crosses below signal line + volume confirmation
            elif trix_aligned[i] < trix_signal_aligned[i] and trix_aligned[i-1] >= trix_signal_aligned[i-1] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: TRIX crosses below signal line
            if trix_aligned[i] < trix_signal_aligned[i] and trix_aligned[i-1] >= trix_signal_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: TRIX crosses above signal line
            if trix_aligned[i] > trix_signal_aligned[i] and trix_aligned[i-1] <= trix_signal_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals