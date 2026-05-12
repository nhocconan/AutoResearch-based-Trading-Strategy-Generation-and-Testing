#!/usr/bin/env python3
# 1d_KAMA_Direction_RSI_Trend_Follow
# Hypothesis: Uses Kaufman Adaptive Moving Average (KAMA) to determine trend direction
# combined with RSI for momentum confirmation on the daily timeframe. Weekly trend filter
# ensures alignment with higher timeframe momentum. Designed to capture sustained moves
# in both bull and bear markets while minimizing whipsaws through adaptive smoothing.
# Low trade frequency expected due to daily timeframe and strict trend alignment.

name = "1d_KAMA_Direction_RSI_Trend_Follow"
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
    volume = prices['volume'].values
    
    # === Weekly Trend Filter ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # === KAMA (10, 2, 30) on daily close ===
    # Efficiency Ratio
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])) > 0)  # placeholder, will compute properly below
    # Recompute volatility as sum of absolute changes over window
    volatility = pd.Series(np.abs(np.diff(close, prepend=close[0]))).rolling(window=30, min_periods=30).sum().values
    er = np.where(volatility > 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA calculation
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === RSI (14) on daily close ===
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(ema_34_1d[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price above KAMA, RSI > 50, and above weekly EMA34
            if (close[i] > kama[i] and 
                rsi[i] > 50 and 
                close[i] > ema_34_1d[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price below KAMA, RSI < 50, and below weekly EMA34
            elif (close[i] < kama[i] and 
                  rsi[i] < 50 and 
                  close[i] < ema_34_1d[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price below KAMA or below weekly EMA34
            if (close[i] < kama[i] or 
                close[i] < ema_34_1d[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price above KAMA or above weekly EMA34
            if (close[i] > kama[i] or 
                close[i] > ema_34_1d[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals