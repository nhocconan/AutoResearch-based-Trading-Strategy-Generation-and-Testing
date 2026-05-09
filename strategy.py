#!/usr/bin/env python3
# 1d KAMA + RSI + Chop Filter
# Hypothesis: On daily timeframe, KAMA identifies trend direction, RSI(14) < 30/ > 70 provides mean-reversion entry,
# and Choppiness Index > 61.8 confirms ranging regime. This combination works in both bull and bear markets by
# fading extremes in ranging conditions while avoiding strong trends. Uses 1w trend filter to avoid counter-trend trades.
# Target: 15-25 trades/year with position size 0.25 for low frequency and high conviction.

name = "1d_KAMA_RSI_ChopFilter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA calculation (ER = 10, fast = 2, slow = 30)
    change = np.abs(np.diff(close, k=10))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # Pad first value
    rsi = np.concatenate([[50], rsi])
    
    # Choppiness Index (14)
    atr = np.zeros_like(close)
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[0], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    sum_atr = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    chop = np.where((max_high - min_low) != 0, 100 * np.log10(sum_atr / (max_high - min_low)) / np.log10(14), 50)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Weekly EMA(20) for trend filter
    close_1w = df_1w['close']
    ema_20_1w = close_1w.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_prev = np.roll(ema_20_1w, 1)
    ema_20_1w_prev[0] = ema_20_1w[0]
    ema_rising_1w = ema_20_1w > ema_20_1w_prev
    ema_falling_1w = ema_20_1w < ema_20_1w_prev
    ema_rising_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_rising_1w)
    ema_falling_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_falling_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or
            np.isnan(ema_rising_1w_aligned[i]) or np.isnan(ema_falling_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price < KAMA (dip in uptrend) AND RSI < 30 AND chop > 61.8 AND weekly uptrend
            if (close[i] < kama[i] and 
                rsi[i] < 30 and 
                chop[i] > 61.8 and 
                ema_rising_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price > KAMA (pullback in downtrend) AND RSI > 70 AND chop > 61.8 AND weekly downtrend
            elif (close[i] > kama[i] and 
                  rsi[i] > 70 and 
                  chop[i] > 61.8 and 
                  ema_falling_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses above KAMA OR RSI > 50
            if (close[i] > kama[i]) or (rsi[i] > 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses below KAMA OR RSI < 50
            if (close[i] < kama[i]) or (rsi[i] < 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals