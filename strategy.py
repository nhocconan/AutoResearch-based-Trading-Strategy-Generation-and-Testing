#!/usr/bin/env python3
"""
12h Weekly Pivot Point Reversal with Volume Confirmation
Strategy: Trade reversals at weekly pivot support/resistance levels.
          Long at S1/S2 with volume confirmation and price > weekly EMA20.
          Short at R1/R2 with volume confirmation and price < weekly EMA20.
          Uses weekly timeframe for pivot calculation and trend filter.
          Designed for low frequency, high-probability reversals in ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points and trend filter (once before loop)
    df_w = get_htf_data(prices, '1w')
    
    # Calculate weekly high, low, close for pivot points
    weekly_high = df_w['high'].values
    weekly_low = df_w['low'].values
    weekly_close = df_w['close'].values
    
    # Calculate pivot points: P = (H + L + C)/3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    # Support and resistance levels
    weekly_s1 = (2 * weekly_pivot) - weekly_high
    weekly_s2 = weekly_pivot - (weekly_high - weekly_low)
    weekly_r1 = (2 * weekly_pivot) - weekly_low
    weekly_r2 = weekly_pivot + (weekly_high - weekly_low)
    
    # Calculate weekly EMA20 for trend filter
    ema_20_w = pd.Series(weekly_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align weekly levels to 12h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_w, weekly_pivot)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_w, weekly_s1)
    weekly_s2_aligned = align_htf_to_ltf(prices, df_w, weekly_s2)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_w, weekly_r1)
    weekly_r2_aligned = align_htf_to_ltf(prices, df_w, weekly_r2)
    ema_20_w_aligned = align_htf_to_ltf(prices, df_w, ema_20_w)
    
    # Volume spike detection (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 100  # need enough history for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(weekly_pivot_aligned[i]) or 
            np.isnan(weekly_s1_aligned[i]) or 
            np.isnan(weekly_s2_aligned[i]) or
            np.isnan(weekly_r1_aligned[i]) or
            np.isnan(weekly_r2_aligned[i]) or
            np.isnan(ema_20_w_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        s1 = weekly_s1_aligned[i]
        s2 = weekly_s2_aligned[i]
        r1 = weekly_r1_aligned[i]
        r2 = weekly_r2_aligned[i]
        ema_20 = ema_20_w_aligned[i]
        
        if position == 0:
            # Long: price near S1 or S2 with volume spike and above weekly EMA20
            if ((abs(price - s1) < 0.005 * price or abs(price - s2) < 0.005 * price) and 
                volume_spike[i] and price > ema_20):
                signals[i] = 0.25
                position = 1
            # Short: price near R1 or R2 with volume spike and below weekly EMA20
            elif ((abs(price - r1) < 0.005 * price or abs(price - r2) < 0.005 * price) and 
                  volume_spike[i] and price < ema_20):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position management
            signals[i] = 0.25
            # Exit: price crosses weekly pivot or below weekly EMA20
            if price > weekly_pivot_aligned[i] or price < ema_20:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position management
            signals[i] = -0.25
            # Exit: price crosses weekly pivot or above weekly EMA20
            if price < weekly_pivot_aligned[i] or price > ema_20:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_WeeklyPivot_Reversal_Volume_EMA20"
timeframe = "12h"
leverage = 1.0