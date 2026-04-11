#!/usr/bin/env python3
# 1d_1w_kama_rsi_trend_v1
# Strategy: 1d KAMA direction + RSI(14) + 1w trend filter
# Timeframe: 1d
# Leverage: 1.0
# Hypothesis: KAMA adapts to market noise, providing reliable trend direction. RSI filters overextended entries.
# 1w EMA acts as higher timeframe trend filter to avoid counter-trend trades. Designed for low frequency
# (10-20 trades/year) to minimize fee drift in both bull and bear markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_kama_rsi_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA(20) for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # KAMA parameters
    er_length = 10
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1) # EMA(30)
    
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=er_length))
    volatility = np.sum(np.abs(np.diff(close)), axis=0) if len(close) > er_length else np.zeros_like(close)
    # Fix volatility calculation using loop for clarity and correctness
    volatility = np.zeros(n)
    for i in range(er_length, n):
        volatility[i] = np.sum(np.abs(np.diff(close[i-er_length+1:i+1])))
    er = np.zeros(n)
    er[er_length:] = change[er_length:] / np.where(volatility[er_length:] == 0, 1, volatility[er_length:])
    
    # Smoothing Constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(ema_20_1w_aligned[i]) or np.isnan(kama[i]) or 
            np.isnan(rsi[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: price relative to 1w EMA20
        uptrend = close[i] > ema_20_1w_aligned[i]
        downtrend = close[i] < ema_20_1w_aligned[i]
        
        # Entry logic: KAMA direction + RSI not extreme + trend alignment
        if (close[i] > kama[i] and  # Price above KAMA = bullish
            rsi[i] < 70 and         # Not overbought
            uptrend and 
            position != 1):
            position = 1
            signals[i] = 0.25
        elif (close[i] < kama[i] and  # Price below KAMA = bearish
              rsi[i] > 30 and         # Not oversold
              downtrend and 
              position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: opposite KAMA cross or RSI extreme
        elif position == 1 and (close[i] < kama[i] or rsi[i] >= 70):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > kama[i] or rsi[i] <= 30):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals