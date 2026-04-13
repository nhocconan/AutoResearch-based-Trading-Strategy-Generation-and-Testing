#!/usr/bin/env python3
"""
6h_1d_Pivot_Momentum_Reversal_v2
Hypothesis: At 6h timeframe, price often reverses from daily pivot levels (S1/S2/R1/R2) with momentum confirmation.
In ranging markets (common in 2025), price oscillates around daily pivots. Uses RSI(2) for mean reversion signals
at extreme pivot levels. Works in both bull/bear by fading extremes at pivot support/resistance.
Target: 15-25 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot points
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily pivot points (standard)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values for today's pivot
    high_prev = np.roll(high_1d, 1)
    low_prev = np.roll(low_1d, 1)
    close_prev = np.roll(close_1d, 1)
    # First day uses same day's values
    high_prev[0] = high_1d[0]
    low_prev[0] = low_1d[0]
    close_prev[0] = close_1d[0]
    
    # Standard pivot point formulas
    pivot = (high_prev + low_prev + close_prev) / 3.0
    r1 = 2 * pivot - low_prev
    s1 = 2 * pivot - high_prev
    r2 = pivot + (high_prev - low_prev)
    s2 = pivot - (high_prev - low_prev)
    
    # Align pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # RSI(2) for mean reversion signals (fast RSI for short-term extremes)
    close_series = pd.Series(close)
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    avg_loss = loss.ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.fillna(50).values  # neutral when undefined
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(rsi_values[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        rsi_val = rsi_values[i]
        
        # Long setup: price near S1/S2 with oversold RSI
        near_s1 = abs(price - s1_aligned[i]) / s1_aligned[i] < 0.005  # within 0.5%
        near_s2 = abs(price - s2_aligned[i]) / s2_aligned[i] < 0.005
        oversold = rsi_val < 20
        
        # Short setup: price near R1/R2 with overbought RSI
        near_r1 = abs(price - r1_aligned[i]) / r1_aligned[i] < 0.005
        near_r2 = abs(price - r2_aligned[i]) / r2_aligned[i] < 0.005
        overbought = rsi_val > 80
        
        long_signal = (near_s1 or near_s2) and oversold
        short_signal = (near_r1 or near_r2) and overbought
        
        if long_signal and position != 1:
            position = 1
            signals[i] = position_size
        elif short_signal and position != -1:
            position = -1
            signals[i] = -position_size
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "6h_1d_Pivot_Momentum_Reversal_v2"
timeframe = "6h"
leverage = 1.0