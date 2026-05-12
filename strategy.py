#!/usr/bin/env python3
"""
6h_Market_Profile_Value_Area_Breakout
Hypothesis: 6h price breaking out of prior 12h Value Area (high-volume range) with volume confirmation
captures institutional breakouts. Works in bull/bear as value areas adapt to volatility.
Value Area defined as 70% of volume within 12h period. Breakout above/below VA with volume > 1.5x avg.
"""

name = "6h_Market_Profile_Value_Area_Breakout"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 12h data for Value Area calculation (call once before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate Value Area (70% volume range) for each 12h bar
    va_high = np.zeros(len(df_12h))
    va_low = np.zeros(len(df_12h))
    
    for i in range(len(df_12h)):
        # Get prices and volumes within this 12h bar
        # Since we don't have tick data, approximate using OHLC
        # Create synthetic price levels between low and high
        n_levels = 20
        price_levels = np.linspace(df_12h['low'].iloc[i], df_12h['high'].iloc[i], n_levels)
        
        # Distribute volume across price levels (simple uniform distribution as approximation)
        # In reality, would use TPO or volume profile, but this approximates the concept
        vol_per_level = df_12h['volume'].iloc[i] / n_levels
        volumes = np.full(n_levels, vol_per_level)
        
        # Find 70% value area
        total_vol = df_12h['volume'].iloc[i]
        target_vol = 0.7 * total_vol
        
        # Sort by price and accumulate volume from high volume areas
        # Simple approach: take range around VWAP
        vwap = np.average(price_levels, weights=volumes)
        va_range = 0.5 * (df_12h['high'].iloc[i] - df_12h['low'].iloc[i])  # 50% of range
        va_high[i] = vwap + va_range
        va_low[i] = vwap - va_range
    
    # Align VA to 6h timeframe
    va_high_aligned = align_htf_to_ltf(prices, df_12h, va_high)
    va_low_aligned = align_htf_to_ltf(prices, df_12h, va_low)

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        if np.isnan(va_high_aligned[i]) or np.isnan(va_low_aligned[i]) or np.isnan(vol_avg_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close breaks above VA High + volume confirmation
            if close[i] > va_high_aligned[i] and volume[i] > vol_avg_20[i] * 1.5:
                signals[i] = 0.25
                position = 1
            # SHORT: Close breaks below VA Low + volume confirmation
            elif close[i] < va_low_aligned[i] and volume[i] > vol_avg_20[i] * 1.5:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close falls back into Value Area or below VA Low
            if close[i] < va_high_aligned[i] or close[i] < va_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close rises back into Value Area or above VA High
            if close[i] > va_low_aligned[i] or close[i] > va_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals