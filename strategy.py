#!/usr/bin/env python3
# 6h_1d_weekly_pivot_breakout_volume_v1
# Hypothesis: 6h price breaking above/below weekly pivot point resistance/support levels
# (R2/S2) with volume confirmation creates high-probability breakout trades.
# Uses weekly timeframe for pivot calculation (stronger support/resistance) and
# 6h for entry timing. Works in both bull/bear markets by trading breakouts in
# direction of prevailing trend. Target: 15-30 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_weekly_pivot_breakout_volume_v1"
timeframe = "6h"
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
    
    # Get weekly data for pivot calculation
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    # Pivot point = (H + L + C) / 3
    pivot = (high_weekly + low_weekly + close_weekly) / 3.0
    # Resistance 1 = (2 * P) - L
    r1 = (2 * pivot) - low_weekly
    # Support 1 = (2 * P) - H
    s1 = (2 * pivot) - high_weekly
    # Resistance 2 = P + (H - L)
    r2 = pivot + (high_weekly - low_weekly)
    # Support 2 = P - (H - L)
    s2 = pivot - (high_weekly - low_weekly)
    # Resistance 3 = H + 2*(P - L)
    r3 = high_weekly + 2 * (pivot - low_weekly)
    # Support 3 = L - 2*(H - P)
    s3 = low_weekly - 2 * (high_weekly - pivot)
    
    # Align weekly pivot levels to 6h timeframe
    pivot_weekly_aligned = align_htf_to_ltf(prices, df_weekly, pivot)
    r1_weekly_aligned = align_htf_to_ltf(prices, df_weekly, r1)
    s1_weekly_aligned = align_htf_to_ltf(prices, df_weekly, s1)
    r2_weekly_aligned = align_htf_to_ltf(prices, df_weekly, r2)
    s2_weekly_aligned = align_htf_to_ltf(prices, df_weekly, s2)
    r3_weekly_aligned = align_htf_to_ltf(prices, df_weekly, r3)
    s3_weekly_aligned = align_htf_to_ltf(prices, df_weekly, s3)
    
    # Volume confirmation: volume > 1.5x average of last 20 periods (5 days)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(pivot_weekly_aligned[i]) or np.isnan(r2_weekly_aligned[i]) or \
           np.isnan(s2_weekly_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below S2 or loses upward momentum
            if close[i] < s2_weekly_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price closes above R2 or loses downward momentum
            if close[i] > r2_weekly_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: price breaks above R2 with volume
            if (close[i] > r2_weekly_aligned[i] and 
                open_prices[i] <= r2_weekly_aligned[i] and  # Ensure breakout happened this bar
                vol_confirm[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below S2 with volume
            elif (close[i] < s2_weekly_aligned[i] and 
                  open_prices[i] >= s2_weekly_aligned[i] and  # Ensure breakdown happened this bar
                  vol_confirm[i]):
                position = -1
                signals[i] = -0.25
    
    return signals