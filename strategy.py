#!/usr/bin/env python3
"""
1h_4d_Time_Price_Oscillator
Hypothesis: Use 4h Time Price Oscillator (TPO) to detect momentum shifts in 1h timeframe.
- Long when: TPO crosses above zero with rising volume and price above 1h EMA20
- Short when: TPO crosses below zero with rising volume and price below 1h EMA20
- Exit when: TPO returns to zero or opposite signal appears
Uses 4h TPO for signal direction (reduces noise), 1h for entry timing and filters.
Target: 15-35 trades/year (60-140 over 4 years) to minimize fee drag.
Works in bull/bear via momentum filter and volume confirmation.
"""

name = "1h_4d_Time_Price_Oscillator"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 4h data for TPO calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # 1h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h TPO: (Close - Low) - (High - Close) = 2*Close - High - Low
    # Normalized by (High - Low) to get -1 to +1 range
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate TPO for each 4h bar
    tpo_4h = np.full_like(close_4h, np.nan)
    for i in range(len(close_4h)):
        if high_4h[i] != low_4h[i]:  # Avoid division by zero
            tpo_4h[i] = (2 * close_4h[i] - high_4h[i] - low_4h[i]) / (high_4h[i] - low_4h[i])
        else:
            tpo_4h[i] = 0.0
    
    # Smooth TPO with 3-period EMA to reduce noise
    tpo_smooth_4h = pd.Series(tpo_4h).ewm(span=3, adjust=False, min_periods=3).mean().values
    
    # Align smoothed TPO to 1h timeframe
    tpo_aligned = align_htf_to_ltf(prices, df_4h, tpo_smooth_4h)
    
    # 1h EMA20 for trend filter
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume confirmation: 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 30  # for EMA20 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(tpo_aligned[i]) or 
            np.isnan(ema20[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume above average
        vol_ok = volume[i] > vol_ma_20[i]
        
        # Price trend filter
        price_above_ema = close[i] > ema20[i]
        price_below_ema = close[i] < ema20[i]
        
        # TPO zero cross signals
        tpo_above_zero = tpo_aligned[i] > 0
        tpo_below_zero = tpo_aligned[i] < 0
        
        if position == 0:
            # Look for entries with TPO momentum + volume + price filter
            if tpo_above_zero and vol_ok and price_above_ema:
                # Long: bullish momentum + volume + price above EMA20
                signals[i] = 0.20
                position = 1
            elif tpo_below_zero and vol_ok and price_below_ema:
                # Short: bearish momentum + volume + price below EMA20
                signals[i] = -0.20
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: TPO returns to zero or turns bearish
                if tpo_aligned[i] <= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            elif position == -1:
                # Exit short: TPO returns to zero or turns bullish
                if tpo_aligned[i] >= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals