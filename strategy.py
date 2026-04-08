#!/usr/bin/env python3
# 6h_1w_1d_camarilla_breakout_v1
# Hypothesis: Uses 1d Camarilla pivot levels on 6h timeframe with weekly trend filter.
# Long when price breaks above R4 with weekly uptrend (price > weekly EMA50).
# Short when price breaks below S4 with weekly downtrend (price < weekly EMA50).
# Exit when price returns to daily pivot (PP) or weekly trend fails.
# Designed for 20-40 trades/year on 6h to avoid fee drag. Works in bull/bear via breakout with trend filter.

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
    
    # 1-day data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1-week data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla pivot levels for previous day
    # PP = (H + L + C) / 3
    # R4 = C + ((H - L) * 1.500)
    # S4 = C - ((H - L) * 1.500)
    pp = (high_1d + low_1d + close_1d) / 3
    r4 = close_1d + (high_1d - low_1d) * 1.5
    s4 = close_1d - (high_1d - low_1d) * 1.5
    
    # Weekly EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 50  # Ensure EMA50 is ready
    
    for i in range(start_idx, n):
        if np.isnan(pp_aligned[i]) or np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(ema50_1w_aligned[i]):
            if position != 0:
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price returns to pivot or weekly trend fails
            if close[i] <= pp_aligned[i] or close[i] < ema50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to pivot or weekly trend fails
            if close[i] >= pp_aligned[i] or close[i] > ema50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above R4 with weekly uptrend
            if close[i] > r4_aligned[i] and close[i] > ema50_1w_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below S4 with weekly downtrend
            elif close[i] < s4_aligned[i] and close[i] < ema50_1w_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals