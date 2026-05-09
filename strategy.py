#!/usr/bin/env python3
"""
Hypothesis: 6h Williams Alligator with 1d trend filter and volume confirmation
Long when price above Alligator teeth (green line) with 1d EMA50 uptrend and volume > 1.5x average
Short when price below Alligator teeth with 1d EMA50 downtrend and volume > 1.5x average
Exit when price crosses the Alligator teeth or trend reverses
Alligator uses smoothed SMAs (Jaw=13, Teeth=8, Lips=5) to avoid whipsaws in ranging markets
Designed to capture trends while filtering noise, suitable for both bull and bear markets
Target: 60-120 total trades over 4 years (15-30/year) with size 0.25
"""

name = "6h_Williams_Alligator_1dEMA50_Volume"
timeframe = "6h"
leverage = 1.0

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
    
    # Calculate Williams Alligator (smoothed SMAs)
    # Jaw: 13-period SMMA, Teeth: 8-period SMMA, Lips: 5-period SMMA
    def smma(data, period):
        """Smoothed Moving Average"""
        if len(data) < period:
            return np.full_like(data, np.nan, dtype=float)
        result = np.full_like(data, np.nan, dtype=float)
        # First value is simple SMA
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (Prev SMMA * (period-1) + Current Close) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    jaw = smma(close, 13)   # Blue line
    teeth = smma(close, 8)  # Red line
    lips = smma(close, 5)   # Green line
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for Alligator calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price above teeth (green line), 1d EMA50 uptrend, volume confirmation
            if (close[i] > teeth[i] and 
                ema50_1d_aligned[i] > ema50_1d_aligned[i-1] and  # EMA rising
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price below teeth, 1d EMA50 downtrend, volume confirmation
            elif (close[i] < teeth[i] and 
                  ema50_1d_aligned[i] < ema50_1d_aligned[i-1] and  # EMA falling
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below teeth or trend reverses
            if (close[i] < teeth[i]) or (ema50_1d_aligned[i] < ema50_1d_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above teeth or trend reverses
            if (close[i] > teeth[i]) or (ema50_1d_aligned[i] > ema50_1d_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals