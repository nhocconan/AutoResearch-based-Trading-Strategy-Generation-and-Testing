#!/usr/bin/env python3
"""
12h Pivot S1/R1 Breakout with Volume and Trend Filter - Optimized
Hypothesis: Daily S1/R1 levels act as strong support/resistance. Breakouts with volume
confirmation and aligned daily trend capture momentum. Reduced frequency by tightening
conditions: volume > 2x average, EMA filter stricter (20/50), and added minimum hold.
Target: 15-25 trades/year per symbol to avoid fee drag.
Works in bull (breakouts continue) and bear (breakdowns continue) markets.
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
    
    # Get daily data for calculations
    df_1d = get_htf_data(prices, '1d')
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily S1, R1, EMAs, and volume average
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = close_1d + (high_1d - low_1d) * 1.1 / 12.0
    s1 = close_1d - (high_1d - low_1d) * 1.1 / 12.0
    
    # Use faster EMAs for trend: 20 and 50
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume average (20-period)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    ema_20_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    start_idx = 50  # need enough for EMA50
    
    for i in range(start_idx, n):
        # Skip if any data unavailable
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_20_aligned[i]) or np.isnan(ema_50_aligned[i]) or
            np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        bars_since_entry += 1
        
        # Trend conditions
        uptrend = ema_20_aligned[i] > ema_50_aligned[i]
        downtrend = ema_20_aligned[i] < ema_50_aligned[i]
        
        # Volume confirmation: stricter threshold
        vol_confirm = volume[i] > 2.0 * vol_ma_aligned[i]
        
        # Breakout conditions
        breakout_up = close[i] > r1_aligned[i]
        breakdown_down = close[i] < s1_aligned[i]
        
        if position == 0:
            # Require minimum 10 bars between entries to reduce frequency
            if bars_since_entry < 10:
                signals[i] = 0.0
                continue
                
            # Long: uptrend + high volume + breakout above R1
            if uptrend and vol_confirm and breakout_up:
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Short: downtrend + high volume + breakdown below S1
            elif downtrend and vol_confirm and breakdown_down:
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        
        elif position == 1:
            # Long exit: trend change OR breakdown below S1 (no volume needed for exit)
            if not uptrend or breakdown_down:
                signals[i] = -0.25  # reverse to short
                position = -1
                bars_since_entry = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend change OR breakout above R1
            if not downtrend or breakout_up:
                signals[i] = 0.25  # reverse to long
                position = 1
                bars_since_entry = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Pivot_S1R1_Breakout_Volume_Optimized"
timeframe = "12h"
leverage = 1.0