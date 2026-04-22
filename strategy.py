#!/usr/bin/env python3
"""
Hypothesis: 4-hour Camarilla Pivot R1/S1 Breakout with 1-day Trend and Volume Confirmation.
Long when price breaks above R1 with 1-day EMA34 rising and volume spike.
Short when price breaks below S1 with 1-day EMA34 falling and volume spike.
Exit when price returns to the pivot point.
Designed for low trade frequency by requiring multiple confirmations.
Works in both bull and bear markets by following the 1-day trend.
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
    
    # Calculate Camarilla pivot levels from previous day
    # Use daily high, low, close from previous day
    # For each 4h bar, we need the previous day's OHLC
    # We'll calculate pivot levels using the daily data
    
    # Load 1d data for pivot calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each day
    # Camarilla: 
    # H = previous day high
    # L = previous day low
    # C = previous day close
    # R1 = C + (H-L)*1.1/12
    # S1 = C - (H-L)*1.1/12
    # Pivot = (H+L+C)/3
    
    # We need to align these levels to the 4h timeframe
    # For each 4h bar, use the previous day's levels
    
    # Extract daily OHLC
    d_high = df_1d['high'].values
    d_low = df_1d['low'].values
    d_close = df_1d['close'].values
    
    # Calculate Camarilla levels
    R1 = d_close + (d_high - d_low) * 1.1 / 12
    S1 = d_close - (d_high - d_low) * 1.1 / 12
    pivot = (d_high + d_low + d_close) / 3
    
    # Align to 4h timeframe (use previous day's levels)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    
    # 1-day EMA34 for trend
    ema34_1d = pd.Series(d_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above R1, EMA34 rising, volume spike
            if (close[i] > R1_aligned[i] and 
                ema34_1d_aligned[i] > ema34_1d_aligned[i-1] and vol_spike):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1, EMA34 falling, volume spike
            elif (close[i] < S1_aligned[i] and 
                  ema34_1d_aligned[i] < ema34_1d_aligned[i-1] and vol_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to pivot point
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses below pivot
                if close[i] < pivot_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price crosses above pivot
                if close[i] > pivot_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Camarilla_R1_S1_Breakout_1dEMA34_Trend_Volume"
timeframe = "4h"
leverage = 1.0