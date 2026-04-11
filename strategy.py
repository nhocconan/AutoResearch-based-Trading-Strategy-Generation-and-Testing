#!/usr/bin/env python3
# 4h_12h_camarilla_pivot_volume_v1
# Strategy: 4-hour Camarilla pivot level bounce with 12-hour volume confirmation
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Camarilla pivot levels (L3, L4, H3, H4) act as strong support/resistance.
# Price approaching these levels with volume > 1.5x 20-period average indicates
# institutional interest. Long near L3/L4 with volume confirmation, short near H3/H4.
# Works in bull by catching bounces off support and in bear by selling resistance.
# Uses 12h volume average to avoid noise and ensure institutional participation.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_camarilla_pivot_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for volume average
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # 12h volume average (20-period)
    vol_12h = df_12h['volume'].values
    vol_avg_20_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_avg_20_12h)
    
    # Daily data for Camarilla pivot calculation (using prior day)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from prior day's OHLC
    # H4 = close + 1.5*(high-low), H3 = close + 1.0*(high-low), L3 = close - 1.0*(high-low), L4 = close - 1.5*(high-low)
    # Using prior day to avoid look-ahead
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Shift by 1 to use prior day's data
    high_1d_prev = np.roll(high_1d, 1)
    low_1d_prev = np.roll(low_1d, 1)
    close_1d_prev = np.roll(close_1d, 1)
    # Set first value to NaN since no prior day
    high_1d_prev[0] = np.nan
    low_1d_prev[0] = np.nan
    close_1d_prev[0] = np.nan
    
    # Calculate Camarilla levels
    camarilla_h4 = close_1d_prev + 1.5 * (high_1d_prev - low_1d_prev)
    camarilla_h3 = close_1d_prev + 1.0 * (high_1d_prev - low_1d_prev)
    camarilla_l3 = close_1d_prev - 1.0 * (high_1d_prev - low_1d_prev)
    camarilla_l4 = close_1d_prev - 1.5 * (high_1d_prev - low_1d_prev)
    
    # Align to 4h timeframe
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after volume average warmup
        # Skip if any required data is invalid
        if (np.isnan(h4_aligned[i]) or np.isnan(h3_aligned[i]) or 
            np.isnan(l3_aligned[i]) or np.isnan(l4_aligned[i]) or
            np.isnan(vol_avg_20_12h_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.5x 12h average
        vol_confirm = volume[i] > 1.5 * vol_avg_20_12h_aligned[i]
        
        # Price proximity to Camarilla levels (within 0.2% of level)
        proximity_threshold = 0.002  # 0.2%
        near_l3 = abs(close[i] - l3_aligned[i]) / l3_aligned[i] < proximity_threshold
        near_l4 = abs(close[i] - l4_aligned[i]) / l4_aligned[i] < proximity_threshold
        near_h3 = abs(close[i] - h3_aligned[i]) / h3_aligned[i] < proximity_threshold
        near_h4 = abs(close[i] - h4_aligned[i]) / h4_aligned[i] < proximity_threshold
        
        # Entry logic: long near support (L3/L4) with volume confirmation
        if (near_l3 or near_l4) and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.25
        # Entry logic: short near resistance (H3/H4) with volume confirmation
        elif (near_h3 or near_h4) and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: opposite signal with volume confirmation
        elif position == 1 and (near_h3 or near_h4) and vol_confirm:
            position = 0
            signals[i] = 0.0
        elif position == -1 and (near_l3 or near_l4) and vol_confirm:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals