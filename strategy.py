#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 1d trend filter and volume confirmation.
# Uses Alligator's Jaw (13-period SMMA), Teeth (8-period SMMA), Lips (5-period SMMA) from 1d timeframe.
# Long when price > Alligator Lips and Teeth > Jaw (bullish alignment) with volume > 1.5x 20-bar avg.
# Short when price < Alligator Lips and Teeth < Jaw (bearish alignment) with volume > 1.5x 20-bar avg.
# Exit when Alligator lines cross (Teeth crosses Jaw) indicating trend weakness.
# Alligator is effective in both trending and ranging markets, providing clear trend direction.
# 1d timeframe filters noise, volume confirmation ensures conviction, reducing false signals.
# Timeframe: 4h, HTF: 1d as per experiment guidelines.

name = "4h_WilliamsAlligator_1dTrend_Volume_v1"
timeframe = "4h"
leverage = 1.0

def _smma(values, period):
    """Smoothed Moving Average (SMMA) - same as Wilder's EMA with alpha=1/period"""
    if len(values) < period:
        return np.full_like(values, np.nan, dtype=float)
    result = np.full_like(values, np.nan, dtype=float)
    # First value is simple average
    result[period-1] = np.mean(values[:period])
    # Subsequent values: SMMA = (prev_smma * (period-1) + current_value) / period
    for i in range(period, len(values)):
        result[i] = (result[i-1] * (period-1) + values[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Alligator and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d Alligator components: Jaw (13), Teeth (8), Lips (5) using SMMA
    close_1d = df_1d['close'].values
    jaw = _smma(close_1d, 13)   # Jaw: 13-period SMMA
    teeth = _smma(close_1d, 8)   # Teeth: 8-period SMMA
    lips = _smma(close_1d, 5)    # Lips: 5-period SMMA
    
    # Align Alligator components to 4h timeframe (completed 1d bar only)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Volume confirmation: volume > 1.5x 20-period average (balanced to avoid overtrading)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for Alligator calculations
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_jaw = jaw_aligned[i]
        curr_teeth = teeth_aligned[i]
        curr_lips = lips_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Bullish alignment: price > Lips and Teeth > Jaw
            if (curr_close > curr_lips and 
                curr_teeth > curr_jaw and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Bearish alignment: price < Lips and Teeth < Jaw
            elif (curr_close < curr_lips and 
                  curr_teeth < curr_jaw and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit when Teeth crosses below Jaw (trend weakness)
            if curr_teeth <= curr_jaw:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when Teeth crosses above Jaw (trend weakness)
            if curr_teeth >= curr_jaw:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals