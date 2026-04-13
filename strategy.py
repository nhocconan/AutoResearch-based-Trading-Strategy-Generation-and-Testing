#!/usr/bin/env python3
"""
Hypothesis: 4h/1d Camarilla pivot reversal with volume confirmation.
Uses daily Camarilla pivot levels (support/resistance) for mean reversion entries.
Long when price touches S3 level with volume confirmation, short when touches R3 level.
Avoids overtrading by requiring both price level touch AND volume spike (>1.5x 20-day avg).
Target: 20-50 trades/year to minimize fee drag while capturing mean reversion in ranging markets.
Works in both bull/bear as it fades extremes at statistical reversal points.
"""

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
    
    # Get 1d data for Camarilla pivots and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # S1 = C - (Range * 1.1/2)
    # S2 = C - (Range * 1.1)
    # S3 = C - (Range * 1.1 * 2)
    # R1 = C + (Range * 1.1/2)
    # R2 = C + (Range * 1.1)
    # R3 = C + (Range * 1.1 * 2)
    
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    daily_range = df_1d['high'] - df_1d['low']
    
    pivot = typical_price.values
    s1 = typical_price - (daily_range * 1.1 / 2)
    s2 = typical_price - (daily_range * 1.1)
    s3 = typical_price - (daily_range * 1.1 * 2)
    r1 = typical_price + (daily_range * 1.1 / 2)
    r2 = typical_price + (daily_range * 1.1)
    r3 = typical_price + (daily_range * 1.1 * 2)
    
    # Align Camarilla levels to 4h timeframe
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3.values)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3.values)
    
    # Calculate volume spike (volume > 1.5x 20-day average)
    vol_ma_20 = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_spike = df_1d['volume'].values > (vol_ma_20 * 1.5)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(s3_aligned[i]) or 
            np.isnan(r3_aligned[i]) or 
            np.isnan(vol_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions: price touches Camarilla S3/R3 with volume confirmation
        # Using 0.1% buffer to avoid whipsaws from exact equality
        buffer = 0.001
        touch_s3 = low[i] <= s3_aligned[i] * (1 + buffer)
        touch_r3 = high[i] >= r3_aligned[i] * (1 - buffer)
        vol_confirm = vol_spike_aligned[i] > 0.5  # True if volume spike
        
        long_entry = touch_s3 and vol_confirm
        short_entry = touch_r3 and vol_confirm
        
        # Exit when price moves back toward midpoint (mean reversion target)
        # Exit long when price reaches S2, exit short when price reaches R2
        # Calculate S2 and R2 for exit
        typical_price_i = (high[i] + low[i] + close[i]) / 3
        daily_range_i = high[i] - low[i]
        s2_exit = typical_price_i - (daily_range_i * 1.1)
        r2_exit = typical_price_i + (daily_range_i * 1.1)
        
        exit_long = position == 1 and close[i] >= s2_exit
        exit_short = position == -1 and close[i] <= r2_exit
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_camarilla_pivot_reversal_v1"
timeframe = "4h"
leverage = 1.0