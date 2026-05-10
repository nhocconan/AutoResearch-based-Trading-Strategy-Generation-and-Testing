#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_Volume
Hypothesis: Camarilla pivot breakout with 1d EMA trend filter and volume confirmation on 12h timeframe.
Uses daily Camarilla levels (R1, S1) from previous day, with trend filter from 1d EMA50 and volume spike confirmation.
Designed to work in both bull and bear markets by following daily trend and using volatility-based entries.
Target: 12-37 trades per year per symbol with strict entry conditions to minimize fee drag.
"""

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "12h"
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
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_50_1d[49] = np.mean(close_1d[:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema_50_1d[i-1]
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate volume SMA(20) for volume confirmation
    vol_sma = np.full(n, np.nan)
    for i in range(20, n):
        vol_sma[i] = np.mean(volume[i-20:i])
    
    # Calculate previous day's Camarilla levels (R1, S1)
    # Camarilla formulas: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    camarilla_r1 = np.full(n, np.nan)
    camarilla_s1 = np.full(n, np.nan)
    
    for i in range(1, n):
        # Use previous day's OHLC to calculate today's levels
        # Since we're on 12h timeframe, we need to align with daily bars
        # We'll calculate based on the 1d data and align it
        pass  # Will calculate in the loop using aligned 1d data
    
    # Get previous day's OHLC from 1d data
    if len(close_1d) >= 2:
        # Calculate Camarilla levels for each 1d bar (using previous day's data)
        camarilla_r1_1d = np.full(len(close_1d), np.nan)
        camarilla_s1_1d = np.full(len(close_1d), np.nan)
        
        for i in range(1, len(close_1d)):
            # Use previous day's OHLC (i-1) to calculate today's (i) Camarilla levels
            prev_high = df_1d['high'].iloc[i-1]
            prev_low = df_1d['low'].iloc[i-1]
            prev_close = df_1d['close'].iloc[i-1]
            
            camarilla_r1_1d[i] = prev_close + 1.1 * (prev_high - prev_low) / 12
            camarilla_s1_1d[i] = prev_close - 1.1 * (prev_high - prev_low) / 12
        
        # Align Camarilla levels to 12h timeframe
        camarilla_r1 = align_htf_to_ltf(prices, df_1d, camarilla_r1_1d)
        camarilla_s1 = align_htf_to_ltf(prices, df_1d, camarilla_s1_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(21, 50)  # Ensure we have enough data for calculations
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_sma[i]) or np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = volume[i] > 1.5 * vol_sma[i]
        
        if position == 0:
            # Long: Break above R1 with uptrend and volume confirmation
            if close[i] > camarilla_r1[i] and close[i] > ema_50_1d_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Break below S1 with downtrend and volume confirmation
            elif close[i] < camarilla_s1[i] and close[i] < ema_50_1d_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Close crosses back below EMA50
            if close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Close crosses back above EMA50
            if close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals