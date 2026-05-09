#!/usr/bin/env python3
"""
1h Camarilla Reversion with 4h Trend Filter
Hypothesis: In ranging markets, price reverts to Camarilla pivot levels (H4/L4).
Use 4h trend to filter direction: only long when 4h trend up at L4, short when 4h trend down at H4.
1h provides precise entry timing. Target: 15-35 trades/year.
Works in bull (trend-following reversals) and bear (mean reversion in ranges).
"""

name = "1h_Camarilla_Reversion_4hTrend"
timeframe = "1h"
leverage = 1.0

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
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA20 for trend filter
    ema20_4h = np.full_like(close_4h, np.nan)
    if len(close_4h) >= 20:
        ema20_4h[19] = np.mean(close_4h[0:20])
        for i in range(20, len(close_4h)):
            ema20_4h[i] = (close_4h[i] * 2 + ema20_4h[i-1] * 18) / 20
    
    # Align 4h EMA20 to 1h timeframe
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    
    # Get 1d data for Camarilla calculation (H4/L4 levels)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla H4 and L4 for each 1d bar
    camarilla_h4_1d = np.full_like(close_1d, np.nan)
    camarilla_l4_1d = np.full_like(close_1d, np.nan)
    
    for i in range(len(df_1d)):
        if not (np.isnan(high_1d[i]) or np.isnan(low_1d[i]) or np.isnan(close_1d[i])):
            range_ = high_1d[i] - low_1d[i]
            camarilla_h4_1d[i] = close_1d[i] + 1.1 * range_ / 2  # H4
            camarilla_l4_1d[i] = close_1d[i] - 1.1 * range_ / 2  # L4
    
    # Align Camarilla levels to 1h timeframe
    camarilla_h4_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4_1d)
    camarilla_l4_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after 4h EMA warmup
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema20_4h_aligned[i]) or 
            np.isnan(camarilla_h4_1d_aligned[i]) or 
            np.isnan(camarilla_l4_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 4h trend
        trend_up = close[i] > ema20_4h_aligned[i]
        
        if position == 0:
            # Enter long at L4 in uptrend (buy the dip in uptrend)
            if trend_up and close[i] <= camarilla_l4_1d_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Enter short at H4 in downtrend (sell the rally in downtrend)
            elif not trend_up and close[i] >= camarilla_h4_1d_aligned[i]:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: trend turns down OR price reaches H4 (take profit)
            if not trend_up or close[i] >= camarilla_h4_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: trend turns up OR price reaches L4 (take profit)
            if trend_up or close[i] <= camarilla_l4_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals