#39
#!/usr/bin/env python3
"""
6h_Supertrend_WeeklyTrend_Filter
Hypothesis: Supertrend (ATR=10, mult=3) on 6h combined with weekly trend filter (price > weekly EMA50) captures major trends while avoiding whipsaws in ranging markets. Weekly filter ensures alignment with higher timeframe momentum, reducing false signals during corrections.
"""

name = "6h_Supertrend_WeeklyTrend_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Supertrend: ATR(10) * 3
    atr_period = 10
    multiplier = 3
    
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[0.0], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Basic Upper and Lower Bands
    hl_avg = (high + low) / 2
    upper_band = hl_avg + (multiplier * atr)
    lower_band = hl_avg - (multiplier * atr)
    
    # Final Supertrend calculation
    final_upper = np.zeros(n)
    final_lower = np.zeros(n)
    supertrend = np.zeros(n)
    trend = np.zeros(n)  # 1 for uptrend, -1 for downtrend
    
    final_upper[:] = upper_band
    final_lower[:] = lower_band
    supertrend[:] = upper_band
    trend[:] = 1
    
    for i in range(1, n):
        if close[i] <= final_upper[i-1]:
            final_upper[i] = upper_band[i]
        else:
            final_upper[i] = final_upper[i-1]
            
        if close[i] >= final_lower[i-1]:
            final_lower[i] = lower_band[i]
        else:
            final_lower[i] = final_lower[i-1]
            
        if trend[i-1] == -1 and close[i] > final_upper[i-1]:
            trend[i] = 1
        elif trend[i-1] == 1 and close[i] < final_lower[i-1]:
            trend[i] = -1
        else:
            trend[i] = trend[i-1]
            
        supertrend[i] = final_lower[i] if trend[i] == 1 else final_upper[i]
    
    # Weekly trend filter: price > weekly EMA50
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_uptrend = df_1w['close'].values > ema_50_1w
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if position == 0:
            # LONG: Supertrend uptrend + weekly uptrend
            if trend[i] == 1 and weekly_uptrend_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Supertrend downtrend + weekly downtrend
            elif trend[i] == -1 and not weekly_uptrend_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Supertrend turns down
            if trend[i] == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Supertrend turns up
            if trend[i] == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals