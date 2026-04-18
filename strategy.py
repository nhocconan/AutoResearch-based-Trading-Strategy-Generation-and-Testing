#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Choppiness_Alligator_MeanReversion_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Choppiness Index
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Choppiness Index (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14)
    chop[0:13] = np.nan
    
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Get weekly data for Alligator
    df_1w = get_htf_data(prices, '1w')
    
    # Alligator lines (Jaw, Teeth, Lips)
    jaw = pd.Series(df_1w['close']).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(df_1w['close']).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(df_1w['close']).rolling(window=5, min_periods=5).mean().shift(3).values
    
    jaw_aligned = align_htf_to_ltf(prices, df_1w, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1w, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1w, lips)
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(chop_aligned[i]) or np.isnan(jaw_aligned[i]) or
            np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        chop_val = chop_aligned[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        
        if position == 0:
            # Mean reversion entry in choppy market (Choppiness > 61.8)
            # Long when price touches or goes below Lips (green line) in chop
            if chop_val > 61.8 and low_val <= lips_val and volume_filter[i]:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            # Short when price touches or goes above Jaw (blue line) in chop
            elif chop_val > 61.8 and high_val >= jaw_val and volume_filter[i]:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
        
        elif position == 1:
            # Long exit: Take profit at Teeth (red line) or stop if chop breaks down
            if high_val >= teeth_val or chop_val < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Take profit at Teeth (red line) or stop if chop breaks down
            if low_val <= teeth_val or chop_val < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals