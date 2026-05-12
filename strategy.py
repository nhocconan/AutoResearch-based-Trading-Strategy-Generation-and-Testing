#!/usr/bin/env python3
name = "4h_KAMA_Direction_1dTrend_Filter"
timeframe = "4h"
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
    volume = prices['volume'].values
    
    # 1d trend filter: EMA34
    df_1d = get_htf_data(prices, '1d')
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # KAMA on 4h: ER = |close - close[9]| / sum(|close - close[-1]|, 9)
    close_series = pd.Series(close)
    change = abs(close_series - close_series.shift(9))
    volatility = abs(close_series - close_series.shift(1)).rolling(9).sum()
    er = change / volatility.replace(0, np.nan)
    sc = (er * 0.59 + 0.01) ** 2  # fast=2/(2+2)=0.667, slow=2/(30+2)=0.064 -> sc=(er*(0.667-0.064)+0.064)^2
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        if not np.isnan(sc.iloc[i]):
            kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # need enough data for 1d EMA34
    
    for i in range(start_idx, n):
        # Skip if 1d trend data not ready
        if np.isnan(ema34_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: KAMA rising + price above KAMA + 1d uptrend
            if (kama[i] > kama[i-1] and  # KAMA rising
                close[i] > kama[i] and   # price above KAMA
                close[i] > ema34_1d_aligned[i]):  # 1d uptrend filter
                signals[i] = 0.25
                position = 1
            # Short: KAMA falling + price below KAMA + 1d downtrend
            elif (kama[i] < kama[i-1] and  # KAMA falling
                  close[i] < kama[i] and   # price below KAMA
                  close[i] < ema34_1d_aligned[i]):  # 1d downtrend filter
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when KAMA falls or price crosses below KAMA
            if (kama[i] < kama[i-1] or close[i] < kama[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when KAMA rises or price crosses above KAMA
            if (kama[i] > kama[i-1] or close[i] > kama[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals