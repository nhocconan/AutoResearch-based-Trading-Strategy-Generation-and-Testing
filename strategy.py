#!/usr/bin/env python3
# 6h_MACD_Histogram_Reversal_WeeklyTrend
# Hypothesis: 6-hour MACD histogram reversal with weekly trend filter
# In bull markets: long when MACD histogram turns positive from negative with weekly uptrend
# In bear markets: short when MACD histogram turns negative from positive with weekly downtrend
# Uses zero-cross of MACD histogram for early momentum shifts, reducing whipsaw vs signal line crosses
# Weekly trend filter ensures alignment with higher timeframe momentum
# Target: 15-35 trades per year (~60-140 over 4 years) with position size 0.25

name = "6h_MACD_Histogram_Reversal_WeeklyTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Load weekly data ONCE for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA20 for trend filter (responsive but smooth)
    ema_20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # MACD components on 6h data
    ema_12 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema_26 = pd.Series(close).ewm(span=26, adjust=False, min_periods=26).mean().values
    macd_line = ema_12 - ema_26
    signal_line = pd.Series(macd_line).ewm(span=9, adjust=False, min_periods=9).mean().values
    macd_hist = macd_line - signal_line
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(26, 9)  # Need 26 for EMA26, 9 for signal line
    
    for i in range(start_idx, n):
        if (np.isnan(ema_20_1w_aligned[i]) or np.isnan(macd_hist[i]) or 
            np.isnan(macd_hist[i-1]) or np.isnan(ema_20_1w_aligned[i-1])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # MACD histogram zero-cross signals
        hist_cross_up = macd_hist[i-1] < 0 and macd_hist[i] >= 0  # Negative to positive
        hist_cross_down = macd_hist[i-1] > 0 and macd_hist[i] <= 0  # Positive to negative
        
        # Weekly trend filter
        weekly_uptrend = close[i] > ema_20_1w_aligned[i]
        weekly_downtrend = close[i] < ema_20_1w_aligned[i]
        
        if position == 0:
            # Long: MACD hist crosses above zero + weekly uptrend
            if hist_cross_up and weekly_uptrend:
                signals[i] = 0.25
                position = 1
            # Short: MACD hist crosses below zero + weekly downtrend
            elif hist_cross_down and weekly_downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: MACD hist crosses below zero or weekly trend turns down
            if hist_cross_down or not weekly_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: MACD hist crosses above zero or weekly trend turns up
            if hist_cross_up or not weekly_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals