#!/usr/bin/env python3
name = "1d_Aroon_Trend_Filtered_1wTrend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def aroon_up(high, period):
    n = len(high)
    up = np.full(n, np.nan)
    for i in range(period - 1, n):
        window = high[i - period + 1:i + 1]
        if len(window) == 0:
            up[i] = np.nan
        else:
            high_idx = np.argmax(window)
            up[i] = ((period - 1 - high_idx) / (period - 1)) * 100
    return up

def aroon_down(low, period):
    n = len(low)
    down = np.full(n, np.nan)
    for i in range(period - 1, n):
        window = low[i - period + 1:i + 1]
        if len(window) == 0:
            down[i] = np.nan
        else:
            low_idx = np.argmin(window)
            down[i] = ((period - 1 - low_idx) / (period - 1)) * 100
    return down

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Weekly trend filter: EMA34
    df_1w = get_htf_data(prices, '1w')
    ema34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Aroon(25) on daily
    aroon_up_val = aroon_up(high, 25)
    aroon_down_val = aroon_down(low, 25)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need enough data for Aroon
    
    for i in range(start_idx, n):
        # Skip if weekly trend data not ready
        if np.isnan(ema34_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Skip if Aroon data not ready
        if np.isnan(aroon_up_val[i]) or np.isnan(aroon_down_val[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Aroon Up > 70 and Aroon Down < 30 (strong uptrend) + weekly uptrend
            if (aroon_up_val[i] > 70 and 
                aroon_down_val[i] < 30 and
                close[i] > ema34_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Aroon Down > 70 and Aroon Up < 30 (strong downtrend) + weekly downtrend
            elif (aroon_down_val[i] > 70 and 
                  aroon_up_val[i] < 30 and
                  close[i] < ema34_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when trend weakens: Aroon Down > 50 or weekly trend turns down
            if (aroon_down_val[i] > 50 or 
                close[i] < ema34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when trend weakens: Aroon Up > 50 or weekly trend turns up
            if (aroon_up_val[i] > 50 or 
                close[i] > ema34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals