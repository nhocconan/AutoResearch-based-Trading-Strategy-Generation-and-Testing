#!/usr/bin/env python3
"""
Hypothesis: 4h Williams Alligator strategy with jaw (13-period SMMA), teeth (8-period SMMA), lips (5-period SMMA).
Long when lips cross above teeth AND teeth above jaw (bullish alignment) AND volume > 1.3x 20-period average.
Short when lips cross below teeth AND teeth below jaw (bearish alignment) AND volume > 1.3x 20-period average.
Exit when Alligator lines re-cross (lips/teeth cross) or opposite alignment occurs.
Uses 1d HTF for Alligator trend confirmation (avoids whipsaws). Target: 75-200 total trades over 4 years (19-50/year).
Williams Alligator identifies trending vs ranging markets - ideal for BTC/ETH in both bull and bear regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(source, length):
    """Smoothed Moving Average (SMMA) - same as RMA/Wilder's"""
    if len(source) < length:
        return np.full_like(source, np.nan, dtype=float)
    result = np.full_like(source, np.nan, dtype=float)
    # First value is simple SMA
    result[length-1] = np.mean(source[:length])
    # Subsequent values: SMMA = (Prev SMMA * (length-1) + Current Price) / length
    for i in range(length, len(source)):
        result[i] = (result[i-1] * (length-1) + source[i]) / length
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d Williams Alligator (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Alligator lines: Jaw (13), Teeth (8), Lips (5) - all SMMA
    jaw = smma(close_1d, 13)
    teeth = smma(close_1d, 8)
    lips = smma(close_1d, 5)
    
    # Align Alligator lines to 4h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # 20-period volume average for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 13)  # volume MA (20), Alligator jaw (13)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        vol_ma_val = vol_ma[i]
        
        # Calculate Alligator alignment for trend direction
        if i >= start_idx + 1:
            jaw_prev = jaw_aligned[i-1]
            teeth_prev = teeth_aligned[i-1]
            lips_prev = lips_aligned[i-1]
            
            # Bullish: lips > teeth > jaw
            bullish_aligned = lips_val > teeth_val and teeth_val > jaw_val
            # Bearish: lips < teeth < jaw
            bearish_aligned = lips_val < teeth_val and teeth_val < jaw_val
            
            # Crossovers: lips/teeth cross
            lips_teeth_bull_cross = lips_val > teeth_val and lips_prev <= teeth_prev
            lips_teeth_bear_cross = lips_val < teeth_val and lips_prev >= teeth_prev
        else:
            bullish_aligned = False
            bearish_aligned = False
            lips_teeth_bull_cross = False
            lips_teeth_bear_cross = False
        
        if position == 0:
            # Long: Bullish alignment AND volume confirmation
            if bullish_aligned and volume[i] > 1.3 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Short: Bearish alignment AND volume confirmation
            elif bearish_aligned and volume[i] > 1.3 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: bearish alignment OR lips/teeth bearish cross
                if bearish_aligned or lips_teeth_bear_cross:
                    exit_signal = True
            elif position == -1:
                # Short exit: bullish alignment OR lips/teeth bullish cross
                if bullish_aligned or lips_teeth_bull_cross:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Williams_Alligator_Alignment_Volume_Confirmation"
timeframe = "4h"
leverage = 1.0