#!/usr/bin/env python3
# 4h_1d_Supertrend_TrendFollowing_With_1d_Supertrend_Filter
# Hypothesis: Combines 4h Supertrend for entry timing with 1d Supertrend as a regime filter to avoid counter-trend trades.
# Uses ATR-based trend following with a higher-timeframe trend filter to improve win rate in both bull and bear markets.
# Designed to generate 20-40 trades per year with controlled risk via ATR-based trailing stops.

name = "4h_1d_Supertrend_TrendFollowing_With_1d_Supertrend_Filter"
timeframe = "4h"
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
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d Supertrend for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # ATR for 1d Supertrend
    tr1_1d = np.abs(high_1d - low_1d)
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    tr_1d[0] = tr1_1d[0]
    atr_1d = pd.Series(tr_1d).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Supertrend parameters
    atr_mult = 3.0
    
    # Calculate upper and lower bands for 1d
    hl2_1d = (high_1d + low_1d) / 2
    upper_band_1d = hl2_1d + (atr_mult * atr_1d)
    lower_band_1d = hl2_1d - (atr_mult * atr_1d)
    
    # Initialize Supertrend arrays
    supertrend_1d = np.full_like(close_1d, np.nan)
    uptrend_1d = np.full_like(close_1d, True)
    
    # Calculate Supertrend for 1d
    for i in range(1, len(close_1d)):
        if np.isnan(upper_band_1d[i]) or np.isnan(lower_band_1d[i]) or np.isnan(atr_1d[i]):
            continue
            
        # Update bands
        if close_1d[i-1] > upper_band_1d[i-1]:
            upper_band_1d[i] = max(upper_band_1d[i], upper_band_1d[i-1])
        else:
            upper_band_1d[i] = upper_band_1d[i]
            
        if close_1d[i-1] < lower_band_1d[i-1]:
            lower_band_1d[i] = min(lower_band_1d[i], lower_band_1d[i-1])
        else:
            lower_band_1d[i] = lower_band_1d[i]
        
        # Determine trend
        if close_1d[i] > upper_band_1d[i-1]:
            uptrend_1d[i] = True
        elif close_1d[i] < lower_band_1d[i-1]:
            uptrend_1d[i] = False
        else:
            uptrend_1d[i] = uptrend_1d[i-1]
            if uptrend_1d[i] and lower_band_1d[i] < lower_band_1d[i-1]:
                lower_band_1d[i] = lower_band_1d[i-1]
            if not uptrend_1d[i] and upper_band_1d[i] > upper_band_1d[i-1]:
                upper_band_1d[i] = upper_band_1d[i-1]
        
        supertrend_1d[i] = lower_band_1d[i] if uptrend_1d[i] else upper_band_1d[i]
    
    # Align 1d Supertrend and trend direction to 4h timeframe
    supertrend_1d_aligned = align_htf_to_ltf(prices, df_1d, supertrend_1d)
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d.astype(float))  # Convert bool to float for alignment
    
    # Calculate 4h Supertrend for entry signals
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    hl2 = (high + low) / 2
    upper_band = hl2 + (atr_mult * atr)
    lower_band = hl2 - (atr_mult * atr)
    
    # Initialize 4h Supertrend
    supertrend = np.full(n, np.nan)
    uptrend = np.full(n, True)
    
    # Calculate Supertrend for 4h
    for i in range(1, n):
        if np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or np.isnan(atr[i]):
            continue
            
        # Update bands
        if close[i-1] > upper_band[i-1]:
            upper_band[i] = max(upper_band[i], upper_band[i-1])
        else:
            upper_band[i] = upper_band[i]
            
        if close[i-1] < lower_band[i-1]:
            lower_band[i] = min(lower_band[i], lower_band[i-1])
        else:
            lower_band[i] = lower_band[i]
        
        # Determine trend
        if close[i] > upper_band[i-1]:
            uptrend[i] = True
        elif close[i] < lower_band[i-1]:
            uptrend[i] = False
        else:
            uptrend[i] = uptrend[i-1]
            if uptrend[i] and lower_band[i] < lower_band[i-1]:
                lower_band[i] = lower_band[i-1]
            if not uptrend[i] and upper_band[i] > upper_band[i-1]:
                upper_band[i] = upper_band[i-1]
        
        supertrend[i] = lower_band[i] if uptrend[i] else upper_band[i]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure sufficient warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if required data is not available
        if (np.isnan(supertrend_1d_aligned[i]) or np.isnan(uptrend_1d_aligned[i]) or
            np.isnan(supertrend[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # Only take trades in direction of 1d Supertrend trend
        trend_filter = uptrend_1d_aligned[i] > 0.5  # True if 1d uptrend
        
        if position == 0:
            # Long: price above 4h Supertrend AND 1d uptrend
            if close[i] > supertrend[i] and trend_filter:
                signals[i] = 0.25
                position = 1
            # Short: price below 4h Supertrend AND 1d downtrend
            elif close[i] < supertrend[i] and not trend_filter:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price closes below 4h Supertrend
            if close[i] < supertrend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price closes above 4h Supertrend
            if close[i] > supertrend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals