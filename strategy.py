#!/usr/bin/env python3
# 12h_RSI2_1dTrend_MeanReversion
# Hypothesis: RSI(2) mean reversion on 12h with 1-day trend filter. Goes long when RSI(2) < 10 and 1-day trend is up (close > EMA50), short when RSI(2) > 90 and 1-day trend is down (close < EMA50). Uses tight RSI thresholds to limit trades and avoid overtrading. Designed to work in both bull and bear markets by aligning with higher timeframe trend.

name = "12h_RSI2_1dTrend_MeanReversion"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend direction
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate RSI(2) on 12h data
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, 0.0), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 2)  # Warmup for 1d EMA and RSI
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(rsi[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter from 1d
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long entry: RSI(2) < 10 AND 1-day uptrend
            if rsi[i] < 10 and uptrend:
                signals[i] = 0.25
                position = 1
            # Short entry: RSI(2) > 90 AND 1-day downtrend
            elif rsi[i] > 90 and downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: RSI(2) > 50 OR 1-day trend turns down
            if rsi[i] > 50 or downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI(2) < 50 OR 1-day trend turns up
            if rsi[i] < 50 or uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals