#!/usr/bin/env python3
"""
1h_4d_Pivot_Reversion
Hypothesis: Mean reversion from daily pivot levels with volume confirmation on 1h timeframe.
Uses daily pivot (R1/S1) as dynamic support/resistance. Mean reversion works in range-bound markets (2025-2026)
and during pullbacks in trending markets. Volume confirmation filters false signals. 1h provides timely entries
while daily pivot provides structure. Target: 20-30 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_daily_pivot(high, low, close):
    """Calculate daily pivot points (standard)."""
    P = (high + low + close) / 3.0
    R1 = 2 * P - low
    S1 = 2 * P - high
    return P, R1, S1

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot levels
    _, R1_1d, S1_1d = calculate_daily_pivot(high_1d, low_1d, close_1d)
    
    # Align daily pivot to 1h timeframe
    R1_1d_aligned = align_htf_to_ltf(prices, df_1d, R1_1d)
    S1_1d_aligned = align_htf_to_ltf(prices, df_1d, S1_1d)
    
    # Volume filter: volume > 1.5x 24-period average (avoid low-volume noise)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean()
    volume_filter = volume > (vol_ma_24 * 1.5)
    
    # RSI filter to avoid overbought/oversold extremes
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.20
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(R1_1d_aligned[i]) or np.isnan(S1_1d_aligned[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(rsi_values[i])):
            signals[i] = 0.0
            continue
        
        # Mean reversion from daily R1/S1 with volume and RSI filters
        long_condition = (close[i] < S1_1d_aligned[i]) and volume_filter[i] and (rsi_values[i] < 35)
        short_condition = (close[i] > R1_1d_aligned[i]) and volume_filter[i] and (rsi_values[i] > 65)
        
        # Exit when price crosses back to pivot level
        exit_long = close[i] > (S1_1d_aligned[i] + (R1_1d_aligned[i] - S1_1d_aligned[i]) * 0.3)
        exit_short = close[i] < (R1_1d_aligned[i] - (R1_1d_aligned[i] - S1_1d_aligned[i]) * 0.3)
        
        if long_condition and position != 1:
            position = 1
            signals[i] = position_size
        elif short_condition and position != -1:
            position = -1
            signals[i] = -position_size
        elif (position == 1 and exit_long) or (position == -1 and exit_short):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "1h_4d_Pivot_Reversion"
timeframe = "1h"
leverage = 1.0