#!/usr/bin/env python3
"""
1d_Keltner_Breakout_1wTrend_RSI
Hypothesis: 1d Keltner Channel breakout in direction of 1w EMA50 trend with RSI filter.
Trades in breakout direction only when aligned with weekly trend and RSI not extreme.
Uses volatility-based channels to adapt to changing market conditions.
Target: 10-25 trades/year.
"""

name = "1d_Keltner_Breakout_1wTrend_RSI"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 50:
        ema_50_1w[49] = np.mean(close_1w[:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_1w)):
            ema_50_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema_50_1w[i-1]
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Keltner Channel (20, 2.0)
    atr_period = 20
    multiplier = 2.0
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = np.full(n, np.nan)
    if n >= atr_period:
        atr[atr_period-1] = np.mean(tr[:atr_period])
        for i in range(atr_period, n):
            atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    ema20 = np.full(n, np.nan)
    if n >= 20:
        ema20[19] = np.mean(close[:20])
        alpha_ema = 2 / (20 + 1)
        for i in range(20, n):
            ema20[i] = alpha_ema * close[i] + (1 - alpha_ema) * ema20[i-1]
    
    kc_upper = ema20 + multiplier * atr
    kc_lower = ema20 - multiplier * atr
    
    # RSI(14)
    rsi_period = 14
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    if n >= rsi_period:
        avg_gain[rsi_period-1] = np.mean(gain[:rsi_period])
        avg_loss[rsi_period-1] = np.mean(loss[:rsi_period])
        for i in range(rsi_period, n):
            avg_gain[i] = (avg_gain[i-1] * (rsi_period-1) + gain[i]) / rsi_period
            avg_loss[i] = (avg_loss[i-1] * (rsi_period-1) + loss[i]) / rsi_period
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20, 14)  # EMA + ATR + RSI warmup
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_1w_aligned[i]) or np.isnan(kc_upper[i]) or np.isnan(kc_lower[i]) or np.isnan(rsi[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above upper Keltner + weekly uptrend + RSI not overbought
            if close[i] > kc_upper[i] and close[i] > ema_50_1w_aligned[i] and rsi[i] < 70:
                signals[i] = 0.25
                position = 1
            # Short: Break below lower Keltner + weekly downtrend + RSI not oversold
            elif close[i] < kc_lower[i] and close[i] < ema_50_1w_aligned[i] and rsi[i] > 30:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Close below EMA20 or weekly trend change
            if close[i] < ema20[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Close above EMA20 or weekly trend change
            if close[i] > ema20[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals