#!/usr/bin/env python3
"""
1d_RSI2_Pullback_1wTrend
Hypothesis: RSI(2) pullbacks on 1d with 1w trend filter.
- Long: RSI(2) < 10 and price > 1w EMA50 (uptrend)
- Short: RSI(2) > 90 and price < 1w EMA50 (downtrend)
- Exit: RSI(2) > 50 for longs, RSI(2) < 50 for shorts
- Designed to capture short-term reversals within the weekly trend
- Target: 5-15 trades/year (20-60 total over 4 years)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Calculate RSI(2)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[1] = gain[1]
    avg_loss[1] = loss[1]
    for i in range(2, n):
        avg_gain[i] = (avg_gain[i-1] * 1 + gain[i]) / 2
        avg_loss[i] = (avg_loss[i-1] * 1 + loss[i]) / 2
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # 1w EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough data for RSI and EMA
    start_idx = max(2, 50) + 1
    
    for i in range(start_idx, n):
        if np.isnan(rsi[i]) or np.isnan(ema50_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: RSI(2) < 10 and price > 1w EMA50 (uptrend)
            if rsi[i] < 10 and close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: RSI(2) > 90 and price < 1w EMA50 (downtrend)
            elif rsi[i] > 90 and close[i] < ema50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: RSI(2) > 50
            if rsi[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI(2) < 50
            if rsi[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_RSI2_Pullback_1wTrend"
timeframe = "1d"
leverage = 1.0