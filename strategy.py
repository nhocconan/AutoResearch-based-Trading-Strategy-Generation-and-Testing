#!/usr/bin/env python3

"""
Hypothesis: Daily Kamas (KAMA) with RSI filter and weekly trend alignment.
Long when KAMA turns up and RSI > 50, short when KAMA turns down and RSI < 50.
Trades only during weekly uptrend/downtrend to avoid whipsaws. Uses KAMA's adaptive smoothing
to reduce noise and capture trends while minimizing false signals in choppy markets.
Designed for low trade frequency (10-30 trades/year) by requiring trend alignment and
momentum confirmation, suitable for 1d timeframe in both bull and bear markets.
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
    volume = prices['volume'].values
    
    # KAMA (Kaufman Adaptive Moving Average) parameters
    fast_ema = 2
    slow_ema = 30
    lookback = 10
    
    # Calculate Efficiency Ratio and smoothing constant
    change = np.abs(np.diff(close, n=lookback))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    er = np.zeros_like(close)
    er[lookback:] = change[lookback:] / volatility[lookback:]
    er[volatility == 0] = 0
    
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI (14-period)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi[:14] = 50  # Neutral before enough data
    
    # Load weekly data for trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # 20-period EMA on weekly close for trend
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if data not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(ema20_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # KAMA direction: slope of KAMA
        kama_up = kama[i] > kama[i-1]
        kama_down = kama[i] < kama[i-1]
        
        if position == 0:
            # Long: KAMA turning up + RSI > 50 + weekly uptrend
            if kama_up and rsi[i] > 50 and ema20_1w_aligned[i] > ema20_1w_aligned[i-1]:
                signals[i] = 0.25
                position = 1
            # Short: KAMA turning down + RSI < 50 + weekly downtrend
            elif kama_down and rsi[i] < 50 and ema20_1w_aligned[i] < ema20_1w_aligned[i-1]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: KAMA direction changes or RSI crosses 50
            exit_signal = False
            
            if position == 1:
                # Exit long: KAMA turns down or RSI < 50
                if kama_down or rsi[i] < 50:
                    exit_signal = True
            else:  # position == -1
                # Exit short: KAMA turns up or RSI > 50
                if kama_up or rsi[i] > 50:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "Daily_KAMA_RSI_WeeklyTrend"
timeframe = "1d"
leverage = 1.0