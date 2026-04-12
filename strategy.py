#!/usr/bin/env python3
"""
1d_1w_camarilla_breakout_v1
Hypothesis: Daily strategy using weekly Camarilla pivot levels with volume confirmation.
Enters long when price breaks above weekly H3 with volume spike; short when breaks below weekly L3 with volume spike.
Uses fixed position sizing (0.25) to minimize churn. Weekly trend filter (price above/below weekly SMA10) to avoid counter-trend trades.
Target: 15-25 trades/year (60-100 total over 4 years) to minimize fee decay while capturing strong weekly moves.
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
    
    # Get weekly data for Camarilla pivot levels and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly SMA10 for trend filter (using weekly close)
    close_1w = df_1w['close'].values
    sma10_1w = np.full_like(close_1w, np.nan)
    for i in range(len(close_1w)):
        if i >= 9:
            sma10_1w[i] = np.mean(close_1w[i-9:i+1])
    
    # Calculate weekly Camarilla pivot levels using PREVIOUS week's data (no look-ahead)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Previous week's values for pivot calculation
    prev_high = np.roll(high_1w, 1)
    prev_low = np.roll(low_1w, 1)
    prev_close = np.roll(close_1w, 1)
    # Set first week's previous values to NaN (no data yet)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Camarilla calculations using previous week's data
    pivot = (prev_high + prev_low + prev_close) / 3
    range_val = prev_high - prev_low
    
    # Weekly Camarilla levels
    h3 = pivot + 1.1 * range_val / 2
    l3 = pivot - 1.1 * range_val / 2
    h4 = pivot + 1.1 * range_val
    l4 = pivot - 1.1 * range_val
    
    # Align weekly levels to daily timeframe
    h3_1d = align_htf_to_ltf(prices, df_1w, h3)
    l3_1d = align_htf_to_ltf(prices, df_1w, l3)
    h4_1d = align_htf_to_ltf(prices, df_1w, h4)
    l4_1d = align_htf_to_ltf(prices, df_1w, l4)
    sma10_1d = align_htf_to_ltf(prices, df_1w, sma10_1w)
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(n):
        if i >= 20:
            vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(h3_1d[i]) or np.isnan(l3_1d[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(sma10_1d[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.8x 20-period average
        volume_filter = volume[i] > vol_ma[i] * 1.8
        
        # Trend filter: price above/below weekly SMA10
        uptrend = close[i] > sma10_1d[i]
        downtrend = close[i] < sma10_1d[i]
        
        # Entry conditions: Weekly Camarilla H3/L3 breakout with volume and trend confirmation
        long_breakout = close[i] > h3_1d[i] and volume_filter and uptrend
        short_breakout = close[i] < l3_1d[i] and volume_filter and downtrend
        
        # Exit conditions: touch opposite H3/L3 level
        long_exit = close[i] < l3_1d[i]
        short_exit = close[i] > h3_1d[i]
        
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

name = "1d_1w_camarilla_breakout_v1"
timeframe = "1d"
leverage = 1.0