#!/usr/bin/env python3
# 12h_Camarilla_Pivot_Bounce_1dTrend_Filter
# Hypothesis: Long when price bounces off Camarilla S1/S2 in uptrend (price > 1d EMA50), short when rejected at R1/R2 in downtrend.
# Uses Camarilla levels from 1d chart with 1d EMA50 trend filter. Works in both bull/bear by following higher timeframe trend.
# Designed for 12-37 trades/year to avoid fee drag.

name = "12h_Camarilla_Pivot_Bounce_1dTrend_Filter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot and EMA
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous day
    typical = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    range_ = df_1d['high'] - df_1d['low']
    
    R4 = typical + range_ * 1.500
    R3 = typical + range_ * 1.250
    R2 = typical + range_ * 1.166
    R1 = typical + range_ * 1.083
    S1 = typical - range_ * 1.083
    S2 = typical - range_ * 1.166
    S3 = typical - range_ * 1.250
    S4 = typical - range_ * 1.500
    
    # Align to 12h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1.values)
    R2_aligned = align_htf_to_ltf(prices, df_1d, R2.values)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1.values)
    S2_aligned = align_htf_to_ltf(prices, df_1d, S2.values)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 1  # Start after first bar to have previous levels
    
    for i in range(start_idx, n):
        if np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or np.isnan(ema_50_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Uptrend: look for bounce at S1 or S2
            if close[i] > ema_50_1d_aligned[i]:
                # Long if price touches/bounces off S1 or S2 with rejection of lower levels
                if ((low[i] <= S1_aligned[i] and close[i] > S1_aligned[i]) or 
                    (low[i] <= S2_aligned[i] and close[i] > S2_aligned[i])):
                    signals[i] = 0.25
                    position = 1
            # Downtrend: look for rejection at R1 or R2
            else:
                # Short if price touches/rejects R1 or R2 with failure to hold higher levels
                if ((high[i] >= R1_aligned[i] and close[i] < R1_aligned[i]) or 
                    (high[i] >= R2_aligned[i] and close[i] < R2_aligned[i])):
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit: price breaks below S1 (failed support) or reaches R1 (profit target)
            if close[i] < S1_aligned[i] or close[i] >= R1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price breaks above R1 (failed resistance) or reaches S1 (profit target)
            if close[i] > R1_aligned[i] or close[i] <= S1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals