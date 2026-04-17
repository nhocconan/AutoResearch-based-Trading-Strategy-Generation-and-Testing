#!/usr/bin/env python3
"""
Hypothesis: 4h timeframe with 1d trend filter using Camarilla pivot levels.
Enter long when price touches S1 level with volume confirmation and above 1d EMA50.
Enter short when price touches R1 level with volume confirmation and below 1d EMA50.
Use mean-reversion at intraday support/resistance with trend filter to avoid counter-trend trades.
Targets 20-40 trades/year to minimize fee drag.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for price action
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h typical price for Camarilla calculation
    typical_price_4h = (high_4h + low_4h + close_4h) / 3.0
    
    # Get 1d data for pivot calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate previous day's Camarilla levels (S1, R1)
    # Using typical price: (H+L+C)/3
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d[0] = np.nan
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    
    # Typical price of previous day
    prev_tp_1d = (prev_high_1d + prev_low_1d + prev_close_1d) / 3.0
    
    # Camarilla levels: S1 = TP - 1.1*(H-L)/12, R1 = TP + 1.1*(H-L)/12
    s1_1d = prev_tp_1d - 1.1 * (prev_high_1d - prev_low_1d) / 12.0
    r1_1d = prev_tp_1d + 1.1 * (prev_high_1d - prev_low_1d) / 12.0
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d data to 4h
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: current volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(s1_1d_aligned[i]) or np.isnan(r1_1d_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price touches or goes below S1 with volume and above EMA50 (uptrend)
            if low[i] <= s1_1d_aligned[i] and volume_filter[i] and close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price touches or goes above R1 with volume and below EMA50 (downtrend)
            elif high[i] >= r1_1d_aligned[i] and volume_filter[i] and close[i] < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price reaches midpoint or shows weakness
            midpoint_1d = (s1_1d_aligned[i] + r1_1d_aligned[i]) / 2.0
            if close[i] >= midpoint_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price reaches midpoint or shows strength
            midpoint_1d = (s1_1d_aligned[i] + r1_1d_aligned[i]) / 2.0
            if close[i] <= midpoint_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1dCamarilla_S1R1_Volume_EMA50"
timeframe = "4h"
leverage = 1.0