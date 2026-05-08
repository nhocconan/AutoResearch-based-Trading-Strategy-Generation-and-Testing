#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_KAMA_Reverse_EMA200_Trend_Filter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA direction on 4h
    close_s = pd.Series(close)
    change = abs(close_s.diff(10))
    volatility = close_s.diff(1).abs().rolling(window=10).sum()
    er = change / volatility.replace(0, 1e-10)
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    kama_up = kama > np.roll(kama, 1)
    kama_down = kama < np.roll(kama, 1)
    
    # EMA200 trend filter from 1d
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    trend_1d = (close_1d > ema200_1d).astype(float)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # Volume filter: above 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma20 * 1.5)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if np.isnan(kama[i]) or np.isnan(trend_1d_aligned[i]) or np.isnan(vol_ma20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long when KAMA turns up in uptrend with volume
            if kama_up[i] and trend_1d_aligned[i] > 0.5 and vol_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short when KAMA turns down in downtrend with volume
            elif kama_down[i] and trend_1d_aligned[i] < 0.5 and vol_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when KAMA turns down or trend fails
            if kama_down[i] or trend_1d_aligned[i] < 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when KAMA turns up or trend fails
            if kama_up[i] or trend_1d_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals