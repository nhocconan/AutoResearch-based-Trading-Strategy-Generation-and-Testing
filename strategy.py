#!/usr/bin/env python3
# 1D_Weekly_HTF_Pullback_LongOnly_v2
# Hypothesis: In multi-year cycles, strong weekly uptrends create reliable daily pullback buying opportunities.
# Uses 1-week EMA21 as trend filter and enters long when price pulls back to daily EMA50 during weekly uptrend.
# Exits when price breaks below daily EMA50 or weekly trend turns down. Designed for low frequency (~10-25 trades/year)
# with position sizing 0.25 to manage drawdown in bear markets like 2022. Works in bull via trend continuation,
# avoids bear markets by staying flat when weekly trend turns down.

name = "1D_Weekly_HTF_Pullback_LongOnly_v2"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Load weekly data ONCE for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    # Weekly EMA21 for trend filter
    ema_21_1w = pd.Series(df_1w['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Daily EMA50 for entry/exit
    ema_50_d = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long
    
    start_idx = 50  # Need 50 periods for daily EMA50
    
    for i in range(start_idx, n):
        if np.isnan(ema_21_1w_aligned[i]) or np.isnan(ema_50_d[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly uptrend filter
        weekly_uptrend = close[i] > ema_21_1w_aligned[i]
        
        if position == 0:
            # Enter long when: weekly uptrend AND price crosses above daily EMA50
            if weekly_uptrend and close[i] > ema_50_d[i] and close[i-1] <= ema_50_d[i-1]:
                signals[i] = 0.25
                position = 1
        elif position == 1:
            # Exit when: price breaks below daily EMA50 OR weekly trend turns down
            if close[i] < ema_50_d[i] or not weekly_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
    
    return signals