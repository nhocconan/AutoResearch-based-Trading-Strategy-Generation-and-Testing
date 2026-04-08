#!/usr/bin/env python3
# 6h_1w_1d_camarilla_breakout_v1
# Hypothesis: Breakout strategy using weekly Camarilla pivot levels (from 1w) and daily trend filter (from 1d).
# Enter long when price breaks above weekly R4 level, price > 1d EMA50, and volume > 1.5x average volume.
# Enter short when price breaks below weekly S4 level, price < 1d EMA50, and volume > 1.5x average volume.
# Exit when price returns to weekly pivot (PP) or trend filter fails.
# Designed for 12-37 trades/year on 6h to avoid fee drag. Works in bull/bear via trend filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_1d_camarilla_breakout_v1"
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
    
    # Weekly Camarilla pivot levels (from 1w)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate pivot points for each weekly bar
    pp_1w = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    r4_1w = pp_1w + (range_1w * 1.5)
    s4_1w = pp_1w - (range_1w * 1.5)
    
    # Align weekly levels to 6h timeframe
    pp_1w_aligned = align_htf_to_ltf(prices, df_1w, pp_1w)
    r4_1w_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    s4_1w_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    
    # 1-day EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume average (20-period) for confirmation
    vol_avg = np.full(n, np.nan)
    for i in range(20, n):
        vol_avg[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(50, 20)  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r4_1w_aligned[i]) or np.isnan(s4_1w_aligned[i]) or 
            np.isnan(pp_1w_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_avg[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirmed = volume[i] > 1.5 * vol_avg[i]
        
        if position == 1:  # Long position
            # Exit: price returns to weekly pivot or trend filter fails
            if close[i] < pp_1w_aligned[i] or close[i] <= ema50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to weekly pivot or trend filter fails
            if close[i] > pp_1w_aligned[i] or close[i] >= ema50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: breakout above weekly R4 with volume and trend filter
            if (close[i] > r4_1w_aligned[i] and 
                close[i] > ema50_1d_aligned[i] and 
                vol_confirmed):
                position = 1
                signals[i] = 0.25
            # Short entry: breakdown below weekly S4 with volume and trend filter
            elif (close[i] < s4_1w_aligned[i] and 
                  close[i] < ema50_1d_aligned[i] and 
                  vol_confirmed):
                position = -1
                signals[i] = -0.25
    
    return signals