#!/usr/bin/env python3
"""
6h_Camarilla_1d_R3_S3_Fade_V1
Hypothesis: Camarilla pivot levels from 1d act as strong reversal zones. Price reaching R3/S3 levels with rejection (wick rejection or close back inside) provides high-probability mean reversion trades. Works in both bull/bear as reversals occur at all market phases. Uses volume confirmation to avoid false breaks.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given period"""
    typical = (high + low + close) / 3
    range_val = high - low
    
    # Camarilla levels
    R4 = close + range_val * 1.1 / 2
    R3 = close + range_val * 1.1 / 4
    R2 = close + range_val * 1.1 / 6
    R1 = close + range_val * 1.1 / 12
    PP = typical
    S1 = close - range_val * 1.1 / 12
    S2 = close - range_val * 1.1 / 6
    S3 = close - range_val * 1.1 / 4
    S4 = close - range_val * 1.1 / 2
    
    return R4, R3, R2, R1, PP, S1, S2, S3, S4

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    # Load 1d data once for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Arrays to store each level
    R4_1d = np.full_like(close_1d, np.nan)
    R3_1d = np.full_like(close_1d, np.nan)
    S3_1d = np.full_like(close_1d, np.nan)
    S4_1d = np.full_like(close_1d, np.nan)
    
    for i in range(len(close_1d)):
        R4, R3, R2, R1, PP, S1, S2, S3, S4 = calculate_camarilla(
            high_1d[i], low_1d[i], close_1d[i]
        )
        R4_1d[i] = R4
        R3_1d[i] = R3
        S3_1d[i] = S3
        S4_1d[i] = S4
    
    # Align Camarilla levels to 6h timeframe
    R3_1d_aligned = align_htf_to_ltf(prices, df_1d, R3_1d)
    S3_1d_aligned = align_htf_to_ltf(prices, df_1d, S3_1d)
    
    # 6h price data
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    volume_6h = prices['volume'].values
    
    # Volume filter: 20-period average volume
    vol_ma = np.full_like(volume_6h, np.nan)
    for i in range(len(volume_6h)):
        if i >= 19:
            vol_ma[i] = np.mean(volume_6h[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if NaN in critical values
        if np.isnan(R3_1d_aligned[i]) or np.isnan(S3_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_6h[i]
        r3_level = R3_1d_aligned[i]
        s3_level = S3_1d_aligned[i]
        vol_ma_val = vol_ma[i]
        volume = volume_6h[i]
        
        # Volume confirmation: current volume > 1.5x average
        volume_ok = volume > vol_ma_val * 1.5
        
        if position == 0:
            # Long setup: price rejects S3 level (wick below, close above) with volume
            if low_6h[i] < s3_level and close_6h[i] > s3_level and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short setup: price rejects R3 level (wick above, close below) with volume
            elif high_6h[i] > r3_level and close_6h[i] < r3_level and volume_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price reaches midpoint between S3 and S2 (or R1) or volume fails
            # Simple exit: return to Pivot Point or opposite S2 level
            # Calculate S2 for exit reference
            if i < len(df_1d):
                # Use current day's S2 approximation
                range_val = high_1d[min(i, len(high_1d)-1)] - low_1d[min(i, len(low_1d)-1)]
                close_val = close_1d[min(i, len(close_1d)-1)]
                s2_level = close_val - range_val * 1.1 / 6
                s2_aligned = align_htf_to_ltf(prices, df_1d, np.full_like(close_1d, s2_level))[i]
                
                if price < s2_aligned:  # Failed to hold above S2
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price reaches midpoint between R3 and R2
            if i < len(df_1d):
                range_val = high_1d[min(i, len(high_1d)-1)] - low_1d[min(i, len(low_1d)-1)]
                close_val = close_1d[min(i, len(close_1d)-1)]
                r2_level = close_val + range_val * 1.1 / 6
                r2_aligned = align_htf_to_ltf(prices, df_1d, np.full_like(close_1d, r2_level))[i]
                
                if price > r2_aligned:  # Failed to hold below R2
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_1d_R3_S3_Fade_V1"
timeframe = "6h"
leverage = 1.0