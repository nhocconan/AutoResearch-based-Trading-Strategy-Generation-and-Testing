#!/usr/bin/env python3
"""
4h_12h_Pivot_R1S1_Breakout_Volume_TrendFilter
Hypothesis: Trade Camarilla pivot breakouts (R1/S1) in direction of 12h EMA(34) trend with volume > 1.5x 24-period average. Uses 12h trend filter to avoid counter-trend trades. Position size 0.25 targeting ~20-30 trades/year to minimize fee drag. Works in bull/bear by trading breakouts with trend alignment and volume confirmation.
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
    
    # Get 12h data for EMA trend filter and Camarilla pivots
    df_12h = get_htf_data(prices, '12h')
    
    # 12h EMA(34) trend filter
    close_12h = df_12h['close'].values
    ema_period = 34
    ema_12h = np.full_like(close_12h, np.nan)
    
    if len(close_12h) >= ema_period:
        ema_12h[ema_period - 1] = np.mean(close_12h[:ema_period])
        for i in range(ema_period, len(close_12h)):
            ema_12h[i] = (close_12h[i] * 2 / (ema_period + 1)) + (ema_12h[i-1] * (ema_period - 1) / (ema_period + 1))
    
    # Align 12h EMA to 4h timeframe
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate Camarilla pivot levels from 12h data
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h_prev = df_12h['close'].values
    
    # Calculate pivot and ranges
    pivot_12h = (high_12h + low_12h + close_12h_prev) / 3.0
    range_12h = high_12h - low_12h
    
    # Camarilla levels
    R1_12h = close_12h_prev + (range_12h * 1.0 / 12.0)
    S1_12h = close_12h_prev - (range_12h * 1.0 / 12.0)
    R2_12h = close_12h_prev + (range_12h * 2.0 / 12.0)
    S2_12h = close_12h_prev - (range_12h * 2.0 / 12.0)
    R3_12h = close_12h_prev + (range_12h * 3.0 / 12.0)
    S3_12h = close_12h_prev - (range_12h * 3.0 / 12.0)
    
    # Align Camarilla levels to 4h timeframe
    R1_12h_aligned = align_htf_to_ltf(prices, df_12h, R1_12h)
    S1_12h_aligned = align_htf_to_ltf(prices, df_12h, S1_12h)
    
    # Volume confirmation: volume > 1.5x 24-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 24
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(ema_period, vol_period)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(R1_12h_aligned[i]) or 
            np.isnan(S1_12h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume and above 12h EMA
            if close[i] > R1_12h_aligned[i] and vol_confirm and close[i] > ema_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume and below 12h EMA
            elif close[i] < S1_12h_aligned[i] and vol_confirm and close[i] < ema_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price closes below S1 or below 12h EMA
            if close[i] < S1_12h_aligned[i] or close[i] < ema_12h_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price closes above R1 or above 12h EMA
            if close[i] > R1_12h_aligned[i] or close[i] > ema_12h_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_12h_Pivot_R1S1_Breakout_Volume_TrendFilter"
timeframe = "4h"
leverage = 1.0