#!/usr/bin/env python3
# Hypothesis: 12h Camarilla R1/S1 breakout with weekly EMA34 trend filter and volume confirmation.
# R1/S1 levels provide tighter risk-reward with more frequent breakouts while maintaining significance.
# Weekly EMA34 filters for long-term trend alignment, reducing counter-trend trades.
# Volume confirmation ensures breakouts have participation.
# Target: 20-40 trades/year to avoid fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 35:  # Need enough for EMA34
        return np.zeros(n)
    
    # Get daily data for Camarilla pivot points
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot points
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Standard Camarilla calculation
    range_1d = high_1d - low_1d
    close_prev = np.roll(close_1d, 1)
    close_prev[0] = close_1d[0]  # First day uses its own close
    
    # Camarilla levels
    r1 = close_prev + 1.1 * range_1d / 12
    s1 = close_prev - 1.1 * range_1d / 12
    
    # Align Camarilla levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Weekly EMA34 for trend filter
    close_1w_series = pd.Series(df_1w['close'].values)
    ema34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume filter: volume > 1.5x 20-period average to avoid noise
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema34_1w_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter
        trend_up = close[i] > ema34_1w_aligned[i]
        trend_down = close[i] < ema34_1w_aligned[i]
        
        # Entry conditions
        # Long: break above R1 with upward trend and volume
        long_breakout = close[i] > r1_aligned[i]
        long_entry = long_breakout and trend_up and volume_filter[i]
        
        # Short: break below S1 with downward trend and volume
        short_breakout = close[i] < s1_aligned[i]
        short_entry = short_breakout and trend_down and volume_filter[i]
        
        # Exit conditions: opposite S1/R1 levels (mean reversion)
        long_exit = close[i] < s1_aligned[i] and position == 1
        short_exit = close[i] > r1_aligned[i] and position == -1
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit:
            signals[i] = 0.0
            position = 0
        elif short_exit:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_Pivot_R1S1_Breakout_1wEMA34_VolumeFilter"
timeframe = "12h"
leverage = 1.0