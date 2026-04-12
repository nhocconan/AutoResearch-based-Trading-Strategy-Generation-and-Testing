#!/usr/bin/env python3
"""
6h_1d_Aggressive_Momentum_Breakout
Hypothesis: Aggressive breakout above 6h Donchian(20) high/low with volume confirmation,
filtered by 1d RSI trend filter (RSI>50 for long, RSI<50 for short). Uses tight stops
via time-based exit (max 3 bars) to limit whipsaw in sideways markets. Designed for
low-frequency, high-conviction trades in both bull and bear regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_Aggressive_Momentum_Breakout"
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
    
    # === 6H DONCHIAN(20) CHANNEL ===
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 1D RSI(14) TREND FILTER ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # === VOLUME CONFIRMATION ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    bars_since_entry = 0
    
    for i in range(20, n):
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(rsi_1d_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            if position != 0:
                bars_since_entry += 1
            continue
        
        # Entry conditions
        long_breakout = (close[i] > high_max[i]) and (vol_ratio[i] > 1.8) and (rsi_1d_aligned[i] > 50)
        short_breakout = (close[i] < low_min[i]) and (vol_ratio[i] > 1.8) and (rsi_1d_aligned[i] < 50)
        
        # Time-based exit: max 3 bars (18 hours)
        time_exit = bars_since_entry >= 3
        
        # Reverse signals for aggressive re-entry
        reverse_long = (position == -1) and long_breakout
        reverse_short = (position == 1) and short_breakout
        
        # Execute trades
        if (long_breakout or reverse_long) and position != 1:
            position = 1
            bars_since_entry = 0
            signals[i] = 0.25
        elif (short_breakout or reverse_short) and position != -1:
            position = -1
            bars_since_entry = 0
            signals[i] = -0.25
        elif time_exit and position != 0:
            position = 0
            bars_since_entry = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
            if position != 0:
                bars_since_entry += 1
    
    return signals