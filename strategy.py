#!/usr/bin/env python3
"""
1d_KAMA_Trend_With_RSI_Momentum
Hypothesis: KAMA (14) provides adaptive trend direction, RSI(2) identifies short-term momentum extremes.
Long when KAMA trending up and RSI<15 (oversold), short when KAMA trending down and RSI>85 (overbought).
Weekly trend filter ensures alignment with higher timeframe trend. Works in both bull and bear markets by
capturing mean-reversion within the trend. Target: 10-25 trades/year per symbol.
"""

name = "1d_KAMA_Trend_With_RSI_Momentum"
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
    
    # KAMA (14) - Kaufman Adaptive Moving Average
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=14))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)
    er = np.zeros(n)
    er[14:] = change[14:] / np.where(volatility[14:] == 0, 1, volatility[14:])
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[13] = close[13]  # seed
    for i in range(14, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI (2) - short-term momentum
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[1] = np.mean(gain[:1])
    avg_loss[1] = np.mean(loss[:1])
    
    for i in range(2, n):
        avg_gain[i] = (avg_gain[i-1] * 1 + gain[i-1]) / 2
        avg_loss[i] = (avg_loss[i-1] * 1 + loss[i-1]) / 2
    
    rs = np.where(avg_loss == 0, 100, avg_gain / avg_loss)
    rsi = 100 - (100 / (1 + rs))
    
    # Weekly trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    ema_20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    uptrend_1w = df_1w['close'].values > ema_20_1w
    downtrend_1w = df_1w['close'].values < ema_20_1w
    uptrend_1w_aligned = align_htf_to_ltf(prices, df_1w, uptrend_1w)
    downtrend_1w_aligned = align_htf_to_ltf(prices, df_1w, downtrend_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Get values
        kama_now = kama[i]
        kama_prev = kama[i-1]
        rsi_now = rsi[i]
        uptrend_htf = uptrend_1w_aligned[i]
        downtrend_htf = downtrend_1w_aligned[i]
        
        if position == 0:
            # LONG: KAMA trending up, RSI oversold, weekly uptrend
            if kama_now > kama_prev and rsi_now < 15 and uptrend_htf:
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA trending down, RSI overbought, weekly downtrend
            elif kama_now < kama_prev and rsi_now > 85 and downtrend_htf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: KAMA turns down or RSI overbought
            if kama_now < kama_prev or rsi_now > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: KAMA turns up or RSI oversold
            if kama_now > kama_prev or rsi_now < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals