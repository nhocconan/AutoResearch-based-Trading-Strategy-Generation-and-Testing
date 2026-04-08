#!/usr/bin/env python3
# 1d_1w_camarilla_pivot_volume_v1
# Hypothesis: 1d price breaking above/below weekly Camarilla pivot point resistance/support levels
# (R1/S1) with volume confirmation creates high-probability breakout trades in both bull and bear markets.
# Weekly pivot points provide strong support/resistance levels that institutions watch.
# Volume confirmation filters false breakouts. Target: 10-25 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_camarilla_pivot_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_prices = prices['open'].values
    
    # Get weekly data for Camarilla pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly Camarilla pivot points
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot point = (H + L + C) / 3
    pivot = (high_1w + low_1w + close_1w) / 3.0
    # Calculate range
    range_1w = high_1w - low_1w
    # Camarilla levels
    r1 = pivot + (range_1w * 1.1 / 12)  # Resistance 1
    s1 = pivot - (range_1w * 1.1 / 12)  # Support 1
    r2 = pivot + (range_1w * 1.1 / 6)   # Resistance 2
    s2 = pivot - (range_1w * 1.1 / 6)   # Support 2
    
    # Align weekly pivot levels to daily timeframe
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2)
    
    # Volume confirmation: volume > 1.8x average of last 15 periods (~3 weeks)
    vol_ma = pd.Series(volume).rolling(window=15, min_periods=15).mean().values
    vol_confirm = volume > vol_ma * 1.8
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(pivot_1w_aligned[i]) or np.isnan(r1_1w_aligned[i]) or \
           np.isnan(s1_1w_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below S1
            if close[i] < s1_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price closes above R1
            if close[i] > r1_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: price breaks above R1 with volume
            if (close[i] > r1_1w_aligned[i] and 
                open_prices[i] <= r1_1w_aligned[i] and  # Ensure breakout happened this bar
                vol_confirm[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below S1 with volume
            elif (close[i] < s1_1w_aligned[i] and 
                  open_prices[i] >= s1_1w_aligned[i] and  # Ensure breakdown happened this bar
                  vol_confirm[i]):
                position = -1
                signals[i] = -0.25
    
    return signals