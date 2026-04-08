#!/usr/bin/env python3
# 12h_1w_1d_cam_pivot_breakout_v1
# Hypothesis: 12h Camarilla pivot breakout with volume confirmation and 1w trend filter.
# Long when price breaks above Camarilla R4 + volume > 1.5x avg + 1w EMA10 up.
# Short when price breaks below Camarilla S4 + volume > 1.5x avg + 1w EMA10 down.
# Exit when price returns to Camarilla pivot (PP) or trend fails.
# Designed for 12-37 trades/year on 12h to avoid fee drag. Works in bull/bear via trend filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_1d_cam_pivot_breakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Camarilla pivot levels from 1d (calculate once)
    lookback = 1
    pivot = np.full(n, np.nan)
    r4 = np.full(n, np.nan)
    s4 = np.full(n, np.nan)
    pp = np.full(n, np.nan)
    
    for i in range(lookback, n):
        h = high[i - lookback]
        l = low[i - lookback]
        c = close[i - lookback]
        pivot[i] = (h + l + c) / 3.0
        pp[i] = pivot[i]
        r4[i] = pivot[i] + 1.5 * (h - l)
        s4[i] = pivot[i] - 1.5 * (h - l)
    
    # 1w EMA10 for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema10_1w = pd.Series(close_1w).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema10_1w_aligned = align_htf_to_ltf(prices, df_1w, ema10_1w)
    
    # Volume average (10-period) for confirmation
    vol_avg = np.full(n, np.nan)
    for i in range(10, n):
        vol_avg[i] = np.mean(volume[i-10:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(20, 10)  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r4[i]) or np.isnan(s4[i]) or np.isnan(pp[i]) or 
            np.isnan(ema10_1w_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirmed = volume[i] > 1.5 * vol_avg[i]
        
        if position == 1:  # Long position
            # Exit: price returns to Camarilla pivot (PP) or trend fails
            if close[i] < pp[i] or close[i] <= ema10_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to Camarilla pivot (PP) or trend fails
            if close[i] > pp[i] or close[i] >= ema10_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: breakout above Camarilla R4 with volume and trend filter
            if (close[i] > r4[i] and 
                vol_confirmed and 
                close[i] > ema10_1w_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: breakdown below Camarilla S4 with volume and trend filter
            elif (close[i] < s4[i] and 
                  vol_confirmed and 
                  close[i] < ema10_1w_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals