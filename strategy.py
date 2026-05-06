#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1-day Williams Alligator with volume confirmation
# Uses 1-day SMoothed Moving Average (SMMA) for Alligator jaws (13), teeth (8), lips (5)
# Long when lips > teeth > jaws with volume > 1.5x average
# Short when lips < teeth < jaws with volume > 1.5x average
# Exit when Alligator lines re-cross (lips crosses teeth) or volume drops
# Target: 15-25 trades per year (60-100 over 4 years) with 0.25 position sizing
# Williams Alligator identifies strong trends; avoids whipsaws in sideways markets
# Works in both bull (trend following) and bear (avoids false signals) regimes

name = "12h_1dWilliamsAlligator_Volume_v1"
timeframe = "12h"
leverage = 1.0

def smma(data, period):
    """Smoothed Moving Average (SMMA) - also called Wilder's Moving Average"""
    if len(data) < period:
        return np.full_like(data, np.nan, dtype=float)
    result = np.full_like(data, np.nan, dtype=float)
    # First value is simple average
    result[period-1] = np.mean(data[:period])
    # Subsequent values: SMMA = (prev_SMMA * (period-1) + current_price) / period
    for i in range(period, len(data)):
        result[i] = (result[i-1] * (period-1) + data[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1-day Williams Alligator components
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Alligator components (using SMMA)
    close_1d = df_1d['close'].values
    jaws = smma(close_1d, 13)   # Blue line
    teeth = smma(close_1d, 8)    # Red line  
    lips = smma(close_1d, 5)     # Green line
    
    # Align Alligator components to 12h timeframe
    jaws_aligned = align_htf_to_ltf(prices, df_1d, jaws)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Volume confirmation: >1.5x 30-period average
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_filter = volume > (1.5 * vol_ma_30)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after volume MA warmup
        # Skip if any critical value is NaN or outside session
        if (np.isnan(jaws_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: Lips > Teeth > Jaws (bullish alignment) + volume
            if lips_aligned[i] > teeth_aligned[i] > jaws_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: Lips < Teeth < Jaws (bearish alignment) + volume
            elif lips_aligned[i] < teeth_aligned[i] < jaws_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Lips crosses below Teeth (trend weakening) OR volume drops
            if lips_aligned[i] < teeth_aligned[i] or not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Lips crosses above Teeth (trend weakening) OR volume drops
            if lips_aligned[i] > teeth_aligned[i] or not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals