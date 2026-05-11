#!/usr/bin/env python3
name = "1d_Keltner_Channel_Breakout_1wTrend"
timeframe = "1d"
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
    
    # Weekly trend filter: EMA200 on weekly close
    df_1w = get_htf_data(prices, '1w')
    ema200_1w = pd.Series(df_1w['close']).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Keltner Channel on daily: EMA20, ATR(10)
    ema20 = pd.Series(close).ewm(span=20, min_periods=20, adjust=False).mean().values
    tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr1 = np.maximum(tr1, np.abs(low[1:] - close[:-1]))
    tr = np.concatenate([[np.nan], tr1])
    atr10 = pd.Series(tr).ewm(span=10, min_periods=10, adjust=False).mean().values
    upper = ema20 + 2.0 * atr10
    lower = ema20 - 2.0 * atr10
    
    position_size = 0.25
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(20, 10, 200)  # EMA20, ATR10, weekly EMA200
    
    for i in range(start_idx, n):
        if np.isnan(ema200_1w_aligned[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(ema20[i]):
            if position == 1:
                signals[i] = 0.0
            elif position == -1:
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        weekly_uptrend = close[i] > ema200_1w_aligned[i]
        weekly_downtrend = close[i] < ema200_1w_aligned[i]
        
        if position == 0:
            if close[i] > upper[i] and weekly_uptrend:
                signals[i] = position_size
                position = 1
            elif close[i] < lower[i] and weekly_downtrend:
                signals[i] = -position_size
                position = -1
        else:
            if position == 1:
                if close[i] < ema20[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                if close[i] > ema20[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size
    
    return signals