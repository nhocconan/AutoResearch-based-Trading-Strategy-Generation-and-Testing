#!/usr/bin/env python3
"""
12h_1d_camarilla_volume_breakout_v1 - 12-hour strategy using daily Camarilla pivot levels
with volume confirmation and trend filter.
Hypothesis: Enter long when price breaks above daily H3 with volume spike and price above EMA50.
Enter short when price breaks below daily L3 with volume spike and price below EMA50.
Uses fixed position sizing (0.25) to minimize churn. Exit on opposite L3/H3 touch.
Target: 15-25 trades/year (60-100 total over 4 years) to minimize fee drift.
Focus on high-probability breakouts in trending markets with volume confirmation.
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
    
    # Get daily data for Camarilla pivot levels and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily EMA50 (using previous day's close to avoid look-ahead)
    close_1d = df_1d['close'].values
    ema50_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 50:
        # Calculate EMA with proper initialization
        alpha = 2.0 / (50 + 1)
        ema50_1d[0] = close_1d[0]
        for i in range(1, len(close_1d)):
            ema50_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema50_1d[i-1]
    
    # Calculate daily Camarilla pivot levels using PREVIOUS day's data (no look-ahead)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values for pivot calculation
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    # Set first day's previous values to NaN (no data yet)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Camarilla calculations using previous day's data
    pivot = (prev_high + prev_low + prev_close) / 3
    range_val = prev_high - prev_low
    
    # Camarilla levels
    h3 = pivot + 1.1 * range_val / 2
    l3 = pivot - 1.1 * range_val / 2
    h4 = pivot + 1.1 * range_val
    l4 = pivot - 1.1 * range_val
    
    # Align daily indicators to 12h timeframe
    h3_12h = align_htf_to_ltf(prices, df_1d, h3)
    l3_12h = align_htf_to_ltf(prices, df_1d, l3)
    h4_12h = align_htf_to_ltf(prices, df_1d, h4)
    l4_12h = align_htf_to_ltf(prices, df_1d, l4)
    ema50_12h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(h3_12h[i]) or np.isnan(l3_12h[i]) or 
            np.isnan(ema50_12h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.8x 20-period average
        volume_filter = volume[i] > vol_ma[i] * 1.8
        
        # Trend filter: price above/below EMA50
        uptrend = close[i] > ema50_12h[i]
        downtrend = close[i] < ema50_12h[i]
        
        # Entry conditions: Camarilla H3/L3 breakout with volume and trend confirmation
        long_breakout = close[i] > h3_12h[i] and volume_filter and uptrend
        short_breakout = close[i] < l3_12h[i] and volume_filter and downtrend
        
        # Exit conditions: touch opposite L3/H3 level
        long_exit = close[i] < l3_12h[i]
        short_exit = close[i] > h3_12h[i]
        
        if long_breakout and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_camarilla_volume_breakout_v1"
timeframe = "12h"
leverage = 1.0