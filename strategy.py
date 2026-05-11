#!/usr/bin/env python3
name = "1d_KAMA_Trend_Filter_RSI"
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
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    # Fast EMA period = 2, Slow EMA period = 30
    fast_period = 2
    slow_period = 30
    
    # Direction = abs(close - close[fast_period])
    change = np.abs(np.diff(close, n=fast_period))
    # Volatility = sum of abs(close - close[1]) over fast_period
    volatility = np.zeros_like(close)
    for i in range(fast_period, len(close)):
        volatility[i] = np.sum(np.abs(np.diff(close[i-fast_period:i+1, 1])))
    
    # Avoid division by zero
    volatility[volatility == 0] = 1e-10
    
    # Efficiency Ratio
    er = change / volatility
    # Smoothing constant
    sc = np.power(er * (2/(fast_period+1) - 2/(slow_period+1)) + 2/(slow_period+1), 2)
    
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    
    # First average
    avg_gain[14] = np.mean(gain[1:15])
    avg_loss[14] = np.mean(loss[1:15])
    
    # Wilder smoothing
    for i in range(15, len(close)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    rsi[:14] = 50  # Neutral before enough data
    
    # Weekly trend filter (1w EMA20)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    weekly_uptrend = close > ema_1w_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(ema_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price > KAMA + RSI > 50 + Weekly uptrend
            if close[i] > kama[i] and rsi[i] > 50 and weekly_uptrend[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price < KAMA + RSI < 50 + Weekly downtrend
            elif close[i] < kama[i] and rsi[i] < 50 and not weekly_uptrend[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price < KAMA or RSI < 40
            if close[i] < kama[i] or rsi[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price > KAMA or RSI > 60
            if close[i] > kama[i] or rsi[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals