#!/usr/bin/env python3
"""
6h_EMA_Crossover_RSI_Filter_WeeklyTrend_v1
Hypothesis: On 6h timeframe, EMA(9)/EMA(21) crossovers filtered by RSI(14) > 50 for longs and < 50 for shorts, with weekly trend confirmation from 1w EMA50, provides robust signals in both bull and bear markets. Weekly trend ensures alignment with higher timeframe momentum, reducing counter-trend whipsaws. RSI filter avoids extreme overbought/oversold conditions that often precede reversals. Discrete sizing (0.0, ±0.25) minimizes fee churn. Targets 12-37 trades per year over 4 years.
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
    
    # Get 1w data for weekly trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # need enough for EMA50
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    close_1w = pd.Series(df_1w['close'].values)
    ema_50_1w = close_1w.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 6h EMAs for crossover signal
    close_s = pd.Series(close)
    ema_9 = close_s.ewm(span=9, adjust=False, min_periods=9).mean().values
    ema_21 = close_s.ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Calculate RSI(14) for filter
    delta = close_s.diff().values
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain_ma = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    loss_ma = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = gain_ma / np.maximum(loss_ma, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA warmup + RSI warmup
    start_idx = max(21, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_9[i]) or np.isnan(ema_21[i]) or np.isnan(rsi[i]) or 
            np.isnan(ema_50_1w_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Weekly trend: price above/below weekly EMA50
        weekly_uptrend = close[i] > ema_50_1w_aligned[i]
        weekly_downtrend = close[i] < ema_50_1w_aligned[i]
        
        if position == 0:
            # Long: EMA9 > EMA21 + RSI > 50 + weekly uptrend
            long_signal = (ema_9[i] > ema_21[i] and 
                          rsi[i] > 50 and 
                          weekly_uptrend)
            
            # Short: EMA9 < EMA21 + RSI < 50 + weekly downtrend
            short_signal = (ema_9[i] < ema_21[i] and 
                           rsi[i] < 50 and 
                           weekly_downtrend)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: EMA9 < EMA21 (trend change) OR RSI < 40 (pullback)
            if ema_9[i] < ema_21[i] or rsi[i] < 40:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: EMA9 > EMA21 (trend change) OR RSI > 60 (bounce)
            if ema_9[i] > ema_21[i] or rsi[i] > 60:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_EMA_Crossover_RSI_Filter_WeeklyTrend_v1"
timeframe = "6h"
leverage = 1.0