#!/usr/bin/env python3
"""
1d_1w_KAMA_Direction_RSI_Trend_Filter
Hypothesis: KAMA adapts to market noise, providing a robust trend filter.
Long when KAMA trends up, RSI > 50 (bullish momentum), and weekly trend confirms.
Short when KAMA trends down, RSI < 50 (bearish momentum), and weekly trend confirms.
This strategy avoids whipsaws in sideways markets by requiring alignment between
daily momentum (RSI), adaptive trend (KAMA), and weekly trend filter.
Target: 10-25 trades/year per symbol.
"""

name = "1d_1w_KAMA_Direction_RSI_Trend_Filter"
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate KAMA (adaptive moving average)
    # ER = |close - close[10]| / sum(|close - close[1]| over 10 periods)
    change = np.abs(np.diff(close, n=10))  # |close[t] - close[t-10]|
    volatility = np.zeros(n)
    for i in range(10, n):
        volatility[i] = np.sum(np.abs(np.diff(close[i-9:i+1], n=1)))  # sum |close[t] - close[t-1]| over 10
    
    er = np.zeros(n)
    er[10:] = change[10:] / np.where(volatility[10:] == 0, 1, volatility[10:])  # avoid div by zero
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # KAMA direction: slope > 0 for uptrend, < 0 for downtrend
    kama_slope = np.zeros(n)
    for i in range(1, n):
        kama_slope[i] = kama[i] - kama[i-1]
    kama_up = kama_slope > 0
    kama_down = kama_slope < 0
    
    # RSI(14) for momentum confirmation
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    for i in range(14, n):
        if i == 14:
            avg_gain[i] = np.mean(gain[1:15])
            avg_loss[i] = np.mean(loss[1:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.zeros(n)
    rsi = np.zeros(n)
    for i in range(14, n):
        if avg_loss[i] == 0:
            rs[i] = 100
        else:
            rs[i] = avg_gain[i] / avg_loss[i]
        rsi[i] = 100 - (100 / (1 + rs[i]))
    
    rsi_bull = rsi > 50
    rsi_bear = rsi < 50
    
    # Weekly trend: 34 EMA
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    uptrend_1w = close_1w > ema_34_1w
    downtrend_1w = close_1w < ema_34_1w
    
    # Align weekly trend to daily
    uptrend_1w_aligned = align_htf_to_ltf(prices, df_1w, uptrend_1w)
    downtrend_1w_aligned = align_htf_to_ltf(prices, df_1w, downtrend_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(35, n):  # Start after warmup for KAMA/RSI
        # Get aligned values for current bar
        kama_up_i = kama_up[i]
        kama_down_i = kama_down[i]
        rsi_bull_i = rsi_bull[i]
        rsi_bear_i = rsi_bear[i]
        uptrend_1w_i = uptrend_1w_aligned[i]
        downtrend_1w_i = downtrend_1w_aligned[i]
        
        if position == 0:
            # LONG: KAMA up, RSI bullish, weekly uptrend
            if kama_up_i and rsi_bull_i and uptrend_1w_i:
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA down, RSI bearish, weekly downtrend
            elif kama_down_i and rsi_bear_i and downtrend_1w_i:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: KAMA turns down or RSI turns bearish
            if not kama_up_i or not rsi_bull_i:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: KAMA turns up or RSI turns bullish
            if not kama_down_i or not rsi_bear_i:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals