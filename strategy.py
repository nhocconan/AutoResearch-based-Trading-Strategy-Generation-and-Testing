#!/usr/bin/env python3
# 4h_12h_camarilla_pivot_volume_v1
# Strategy: 4h Camarilla pivot reversal with 12h volume confirmation
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Camarilla pivot levels (L3, L3, H3, H4) act as strong support/resistance.
# Price rejection at these levels with high volume indicates institutional interest.
# Works in both bull and bear markets as it captures mean reversion at key levels.
# Low trade frequency expected due to strict pivot + volume confirmation.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_camarilla_pivot_volume_v1"
timeframe = "4h"
leverage = 1.0

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given high, low, close"""
    range_val = high - low
    if range_val <= 0:
        return close, close, close, close, close, close
    close_val = close
    L3 = close_val - (range_val * 1.1 / 4)
    L4 = close_val - (range_val * 1.1 / 2)
    H3 = close_val + (range_val * 1.1 / 4)
    H4 = close_val + (range_val * 1.1 / 2)
    return L3, L4, H3, H4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla levels from 12h data
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Initialize arrays for Camarilla levels
    L3_12h = np.full_like(close_12h, np.nan)
    L4_12h = np.full_like(close_12h, np.nan)
    H3_12h = np.full_like(close_12h, np.nan)
    H4_12h = np.full_like(close_12h, np.nan)
    
    # Calculate Camarilla for each 12h bar
    for i in range(len(close_12h)):
        L3, L4, H3, H4 = calculate_camarilla(high_12h[i], low_12h[i], close_12h[i])
        L3_12h[i] = L3
        L4_12h[i] = L4
        H3_12h[i] = H3
        H4_12h[i] = H4
    
    # Align Camarilla levels to 4h timeframe (no extra delay needed as these are based on completed 12h bar)
    L3_12h_aligned = align_htf_to_ltf(prices, df_12h, L3_12h)
    L4_12h_aligned = align_htf_to_ltf(prices, df_12h, L4_12h)
    H3_12h_aligned = align_htf_to_ltf(prices, df_12h, H3_12h)
    H4_12h_aligned = align_htf_to_ltf(prices, df_12h, H4_12h)
    
    # 12h volume average (20-period) for confirmation
    volume_12h = df_12h['volume'].values
    vol_avg_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_avg_20_12h)
    
    # Align raw 12h volume for confirmation
    vol_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(L3_12h_aligned[i]) or np.isnan(L4_12h_aligned[i]) or 
            np.isnan(H3_12h_aligned[i]) or np.isnan(H4_12h_aligned[i]) or
            np.isnan(vol_avg_20_12h_aligned[i]) or np.isnan(vol_12h_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current 12h volume > 1.5x 20-period average
        vol_confirm = vol_12h_aligned[i] > 1.5 * vol_avg_20_12h_aligned[i]
        
        # Price levels
        L3 = L3_12h_aligned[i]
        L4 = L4_12h_aligned[i]
        H3 = H3_12h_aligned[i]
        H4 = H4_12h_aligned[i]
        
        # Entry conditions with price rejection logic
        # Long: Price rejects L4 (moves above L4 after being below) OR bounces from L3
        long_signal = False
        if i > 0:  # Need previous close for rejection detection
            # Rejection above L4: was below or at L4, now above L4
            if close[i-1] <= L4 and close[i] > L4:
                long_signal = True
            # Bounce from L3: was at or above L3, now above L3 (but not strong rejection)
            elif close[i-1] >= L3 and close[i] > L3 and close[i] < (L3 + H3)/2:  # In lower half
                long_signal = True
        
        # Short: Price rejects H3 (moves below H3 after being above) OR rejection from H4
        short_signal = False
        if i > 0:
            # Rejection below H3: was above or at H3, now below H3
            if close[i-1] >= H3 and close[i] < H3:
                short_signal = True
            # Rejection from H4: was at or above H4, now below H4
            elif close[i-1] >= H4 and close[i] < H4:
                short_signal = True
        
        # Execute signals with volume confirmation
        if long_signal and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: Price moves to opposite side of midpoint (mean reversion)
        elif position == 1 and close[i] > (L3 + H3)/2:  # Move to upper half
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] < (L3 + H3)/2:  # Move to lower half
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals