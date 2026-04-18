#!/usr/bin/env python3
"""
1h_EMA_Retracement_With_4hTrend_and_1dVol
Hypothesis: In trending markets (4h EMA20), price retraces to the 20 EMA on 1h and resumes trend.
Enter long when price touches 1h EMA20 from below in 4h uptrend, short when from above in 4h downtrend.
Use 1d volume filter to avoid low-activity periods. Works in bull/bear by following 4h trend.
Target: 15-35 trades/year on 1h with strict entry conditions.
"""

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
    volume = prices['volume'].values
    
    # 1h EMA20 for entry
    ema20 = np.full(n, np.nan)
    k = 2 / (20 + 1)
    for i in range(20, n):
        if i == 20:
            ema20[i] = np.mean(close[0:21])
        else:
            ema20[i] = close[i] * k + ema20[i-1] * (1 - k)
    
    # 4h EMA20 trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema20_4h = np.full(len(close_4h), np.nan)
    for i in range(20, len(close_4h)):
        if i == 20:
            ema20_4h[i] = np.mean(close_4h[0:21])
        else:
            ema20_4h[i] = close_4h[i] * k + ema20_4h[i-1] * (1 - k)
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    
    # 1d volume MA filter (avoid low volume)
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    vol_ma_1d = np.full(len(volume_1d), np.nan)
    for i in range(20, len(volume_1d)):
        vol_ma_1d[i] = np.mean(volume_1d[i-20:i])
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # EMA20 needs 20 bars
    
    for i in range(start_idx, n):
        if (np.isnan(ema20[i]) or np.isnan(ema20_4h_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        vol_filter = volume[i] > vol_ma_1d_aligned[i]  # above average 1d volume
        
        if position == 0 and in_session and vol_filter:
            # Long: price touches EMA20 from below in 4h uptrend
            if (low[i] <= ema20[i] and close[i] > ema20[i] and 
                close[i] > ema20_4h_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short: price touches EMA20 from above in 4h downtrend
            elif (high[i] >= ema20[i] and close[i] < ema20[i] and 
                  close[i] < ema20_4h_aligned[i]):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: price closes below EMA20 or 4h trend turns down
            if (close[i] < ema20[i] or close[i] < ema20_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: price closes above EMA20 or 4h trend turns up
            if (close[i] > ema20[i] or close[i] > ema20_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_EMA_Retracement_With_4hTrend_and_1dVol"
timeframe = "1h"
leverage = 1.0