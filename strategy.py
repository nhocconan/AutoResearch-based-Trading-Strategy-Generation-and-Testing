#!/usr/bin/env python3
# 6h_weekly_pivot_breakout_volume_v1
# Hypothesis: Breakout trades in direction of weekly pivot trend with volume confirmation.
# Long when price breaks above weekly R1 with volume > 1.5x average and price > weekly pivot.
# Short when price breaks below weekly S1 with volume > 1.5x average and price < weekly pivot.
# Uses weekly pivot levels for trend direction and 6h price action for entry timing.
# Target: 50-150 total trades over 4 years (~12-37/year).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_pivot_breakout_volume_v1"
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
    
    # Get weekly data ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) == 0:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using previous week's OHLC)
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    weekly_open = df_weekly['open'].values
    
    # Pivot point = (H + L + C) / 3
    pp = (weekly_high + weekly_low + weekly_close) / 3.0
    # R1 = (2 * PP) - L
    r1 = (2 * pp) - weekly_low
    # S1 = (2 * PP) - H
    s1 = (2 * pp) - weekly_high
    
    # Align weekly levels to 6h timeframe (wait for weekly bar to close)
    pp_aligned = align_htf_to_ltf(prices, df_weekly, pp)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1)
    
    # Average volume for confirmation (24-period = 4 days on 6h)
    avg_volume = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(avg_volume[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below weekly pivot or volume drops below average
            if close[i] < pp_aligned[i] or volume[i] < avg_volume[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above weekly pivot or volume drops below average
            if close[i] > pp_aligned[i] or volume[i] < avg_volume[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average volume
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            # Breakout entries in direction of weekly pivot trend
            if close[i] > r1_aligned[i] and close[i] > pp_aligned[i] and volume_ok:
                # Additional confirmation: previous close was at or below R1
                if i > 0 and close[i-1] <= r1_aligned[i-1]:
                    position = 1
                    signals[i] = 0.25
            elif close[i] < s1_aligned[i] and close[i] < pp_aligned[i] and volume_ok:
                # Additional confirmation: previous close was at or above S1
                if i > 0 and close[i-1] >= s1_aligned[i-1]:
                    position = -1
                    signals[i] = -0.25
    
    return signals