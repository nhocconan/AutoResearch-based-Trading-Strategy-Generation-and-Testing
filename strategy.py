#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h_1w_1d_alligator_elder_ray_v1
# Combines weekly Alligator (Jaw/Teeth/Lips from SMA13,8,5) with daily Elder Ray (Bull/Bear Power).
# Long when: price > weekly Teeth AND daily Bull Power > 0 AND Bear Power < 0 (bullish alignment).
# Short when: price < weekly Teeth AND daily Bear Power < 0 AND Bull Power < 0 (bearish alignment).
# Uses 13-period EMA for smoothing and volatility filter (ATR > 0.5*ATR(50)) to avoid chop.
# Designed for low trade frequency (target: 15-35 trades/year) with strong trend alignment.

name = "6h_1w_1d_alligator_elder_ray_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for Alligator (Jaw, Teeth, Lips)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 13:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Alligator lines: Jaw (SMA13), Teeth (SMA8), Lips (SMA5)
    jaw = pd.Series(close_1w).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(close_1w).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(close_1w).rolling(window=5, min_periods=5).mean().values
    
    # Align Alligator lines to 6h
    jaw_aligned = align_htf_to_ltf(prices, df_1w, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1w, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1w, lips)
    
    # Get daily data for Elder Ray (Bull Power, Bear Power) and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily EMA13 for Elder Ray calculation
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13
    bull_power = high_1d - ema13_1d
    # Bear Power = Low - EMA13
    bear_power = low_1d - ema13_1d
    
    # Align Elder Ray components to 6h
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Daily ATR for volatility filter
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_ma_50 = pd.Series(atr_1d).rolling(window=50, min_periods=50).mean().values
    atr_ma_50_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(atr_ma_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: avoid extremely low volatility (chop)
        if atr_1d[i] < 0.5 * atr_ma_50_aligned[i]:
            # Hold current position if volatility too low
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Long conditions: bullish alignment across timeframes
        bullish = (close[i] > teeth_aligned[i] and 
                   bull_power_aligned[i] > 0 and 
                   bear_power_aligned[i] < 0)
        
        # Short conditions: bearish alignment across timeframes
        bearish = (close[i] < teeth_aligned[i] and 
                   bear_power_aligned[i] < 0 and 
                   bull_power_aligned[i] < 0)
        
        # Enter long on bullish alignment
        if bullish and position != 1:
            position = 1
            signals[i] = 0.25
        # Enter short on bearish alignment
        elif bearish and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit when price crosses Teeth (Alligator reversal signal)
        elif position == 1 and close[i] <= teeth_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] >= teeth_aligned[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals