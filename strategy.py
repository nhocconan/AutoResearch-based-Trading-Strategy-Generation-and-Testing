#!/usr/bin/env python3
# 4h_Camarilla_Pivot_R1_S1_Breakout_Volume_Confirmation
# Hypothesis: Camarilla pivot levels from daily chart act as strong support/resistance.
# Price breaking above R1 with volume confirmation signals bullish momentum.
# Price breaking below S1 with volume confirmation signals bearish momentum.
# Uses 4h chart for execution, 1d for pivot levels. Works in both bull and bear markets
# by trading breakouts in direction of the break. Volume filter reduces false signals.
# Target: 20-50 trades per year to avoid fee drag.

name = "4h_Camarilla_Pivot_R1_S1_Breakout_Volume_Confirmation"
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
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day's OHLC
    # R1 = C + (H-L)*1.1/12
    # S1 = C - (H-L)*1.1/12
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Shift by 1 to use previous day's data (avoid look-ahead)
    prev_close = np.concatenate([[np.nan], close_1d[:-1]])
    prev_high = np.concatenate([[np.nan], high_1d[:-1]])
    prev_low = np.concatenate([[np.nan], low_1d[:-1]])
    
    # Calculate Camarilla levels
    R1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    S1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Align to 4h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = np.zeros_like(volume)
    for i in range(20, len(volume)):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    volume_confirmed = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 1)  # Ensure we have enough data for volume MA
    
    for i in range(start_idx, n):
        # Skip if pivot levels are not available (first day)
        if np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 with volume confirmation
            if close[i] > R1_aligned[i] and volume_confirmed[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume confirmation
            elif close[i] < S1_aligned[i] and volume_confirmed[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below S1 (failed breakout) or reverse signal
            if close[i] < S1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above R1 (failed breakout) or reverse signal
            if close[i] > R1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals