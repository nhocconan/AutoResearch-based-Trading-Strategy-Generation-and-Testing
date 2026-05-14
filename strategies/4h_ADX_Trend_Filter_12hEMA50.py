# !/usr/bin/env python3
# 4h_ADX_Trend_Filter_12hEMA50
# Hypothesis: Uses ADX(14) on 4h to identify strong trends and 12h EMA(50) as trend direction filter.
# Enters long when ADX > 25 (strong trend) and price above 12h EMA50 (uptrend).
# Enters short when ADX > 25 and price below 12h EMA50 (downtrend).
# Exits when ADX < 20 (weak trend) or trend direction changes.
# Designed for 25-40 trades/year on 4h to avoid overtrading and work in both bull and bear markets.

name = "4h_ADX_Trend_Filter_12hEMA50"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # ADX(14) calculation
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0
        minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0
        
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
    atr = np.zeros(n)
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    # Initial values
    atr[13] = np.mean(tr[1:14])
    plus_dm_sum = np.sum(plus_dm[1:14])
    minus_dm_sum = np.sum(minus_dm[1:14])
    
    for i in range(14, n):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
        plus_dm_sum = plus_dm_sum - (plus_dm_sum / 14) + plus_dm[i]
        minus_dm_sum = minus_dm_sum - (minus_dm_sum / 14) + minus_dm[i]
        plus_di[i] = 100 * plus_dm_sum / atr[i] if atr[i] != 0 else 0
        minus_di[i] = 100 * minus_dm_sum / atr[i] if atr[i] != 0 else 0
    
    dx = np.zeros(n)
    adx = np.zeros(n)
    
    for i in range(14, n):
        di_sum = plus_di[i] + minus_di[i]
        dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum if di_sum != 0 else 0
    
    # Smooth DX to get ADX
    adx[27] = np.mean(dx[14:28])  # First ADX value after 2*period
    for i in range(28, n):
        adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # 12h EMA(50) for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure sufficient warmup for ADX and EMA
    
    for i in range(start_idx, n):
        if np.isnan(adx[i]) or np.isnan(ema_50_12h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Strong trend (ADX > 25) and price above 12h EMA50 (uptrend)
            if adx[i] > 25 and close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Strong trend (ADX > 25) and price below 12h EMA50 (downtrend)
            elif adx[i] > 25 and close[i] < ema_50_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Weak trend (ADX < 20) or trend reversal (price below EMA)
            if adx[i] < 20 or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Weak trend (ADX < 20) or trend reversal (price above EMA)
            if adx[i] < 20 or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals