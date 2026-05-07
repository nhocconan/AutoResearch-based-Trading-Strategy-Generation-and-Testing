#!/usr/bin/env python3
"""
6h_Pressure_Pattern_Reversal_v1
Hypothesis: Combines multi-timeframe pressure analysis with mean reversion at extreme levels.
Uses 12h pressure index (close position within 12h range) for regime detection and
6h Williams %R for entry timing. In high-pressure regimes (trending), we fade extremes.
In low-pressure regimes (ranging), we follow momentum. Targets 15-30 trades/year.
Works in both bull/bear by adapting to market regime via pressure dynamics.
"""

name = "6h_Pressure_Pattern_Reversal_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 12h data for pressure index calculation (regime filter)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    
    # Calculate 12h Pressure Index: (close - low) / (high - low) averaged over 14 periods
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Avoid division by zero
    range_12h = high_12h - low_12h
    range_12h = np.where(range_12h == 0, 1e-10, range_12h)
    pressure_12h = (close_12h - low_12h) / range_12h
    
    # Smooth pressure index with 14-period average
    pressure_ma = pd.Series(pressure_12h).rolling(window=14, min_periods=14).mean().values
    pressure_ma_aligned = align_htf_to_ltf(prices, df_12h, pressure_ma)
    
    # Calculate 6h Williams %R for entry timing (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    wr_range = highest_high - lowest_low
    wr_range = np.where(wr_range == 0, 1e-10, wr_range)
    williams_r = -100 * (highest_high - close) / wr_range
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):
        # Skip if any critical value is NaN
        if (np.isnan(pressure_ma_aligned[i]) or np.isnan(williams_r[i]) or
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        pressure = pressure_ma_aligned[i]
        wr = williams_r[i]
        
        if position == 0:
            # High pressure regime (>0.7) = trending: fade extreme Williams %R
            if pressure > 0.7:
                if wr <= -80:  # Oversold in uptrend -> long
                    signals[i] = 0.25
                    position = 1
                elif wr >= -20:  # Overbought in uptrend -> short
                    signals[i] = -0.25
                    position = -1
            # Low pressure regime (<0.3) = ranging: momentum follow
            elif pressure < 0.3:
                if wr >= -20 and close[i] > close[i-1]:  # Overbought with upward momentum
                    signals[i] = 0.25
                    position = 1
                elif wr <= -80 and close[i] < close[i-1]:  # Oversold with downward momentum
                    signals[i] = -0.25
                    position = -1
            # Medium pressure: no trade
        elif position == 1:
            # Long exit: Williams %R returns from extreme OR pressure shifts significantly
            if wr >= -50 or pressure < 0.4:  # Return from oversold or pressure drops
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R returns from extreme OR pressure shifts significantly
            if wr <= -50 or pressure > 0.6:  # Return from overbought or pressure rises
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals