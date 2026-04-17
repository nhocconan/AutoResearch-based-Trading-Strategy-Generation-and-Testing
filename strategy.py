#!/usr/bin/env python3
"""
12h_MultiTimeframe_StructureBreakout_v1
12-hour strategy combining weekly Bollinger Band breakouts with daily volume confirmation.
Designed to capture structural breaks in both bull and bear markets with low trade frequency.
Target: 50-150 total trades over 4 years (12-37/year).
"""

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
    
    # === Weekly Bollinger Bands (20-period, 2 std) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    bb_middle = pd.Series(close_1w).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close_1w).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    
    # Align weekly BB to 12h timeframe (wait for weekly close)
    bb_upper_aligned = align_htf_to_ltf(prices, df_1w, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1w, bb_lower)
    bb_middle_aligned = align_htf_to_ltf(prices, df_1w, bb_middle)
    
    # === Daily volume confirmation (20-period average) ===
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(bb_upper_aligned[i]) or 
            np.isnan(bb_lower_aligned[i]) or 
            np.isnan(bb_middle_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume confirmation: current daily volume > 1.5x 20-day average
        # Need to get current day's volume from 1d data aligned to current 12h bar
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        vol_confirmed = vol_1d_current > 1.5 * vol_ma_1d_aligned[i]
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above weekly BB upper with volume confirmation
            if (close[i] > bb_upper_aligned[i] and vol_confirmed):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below weekly BB lower with volume confirmation
            elif (close[i] < bb_lower_aligned[i] and vol_confirmed):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: price returns to weekly BB middle
        elif position == 1:
            # Exit long: price crosses below weekly BB middle
            if close[i] < bb_middle_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above weekly BB middle
            if close[i] > bb_middle_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_MultiTimeframe_StructureBreakout_v1"
timeframe = "12h"
leverage = 1.0