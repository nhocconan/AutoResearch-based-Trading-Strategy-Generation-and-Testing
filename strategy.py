#!/usr/bin/env python3
# 1d_1w_kama_rsi_trend_v1
# Strategy: 1d KAMA with RSI filter and weekly trend filter
# Timeframe: 1d
# Leverage: 1.0
# Hypothesis: KAMA adapts to market noise, reducing false signals in choppy markets. RSI filters overextended entries. Weekly trend filter ensures alignment with higher timeframe direction. Designed for low trade frequency (~15-25/year) to minimize fee drag and work in both bull and bear markets via adaptive trend following.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_kama_rsi_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA(34) for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # KAMA (Adaptive Moving Average)
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close[t] - close[t-10]|
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # sum |close[t] - close[t-1]| over 10 periods
    # Fix array lengths
    change = np.concatenate([np.full(10, np.nan), change])
    volatility = np.concatenate([np.full(10, np.nan), volatility])
    er = np.divide(change, volatility, out=np.full_like(change, np.nan), where=volatility!=0)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[0] = close[0]
    for i in range(1, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # RSI(14) for momentum filter
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(ema_34_1w_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Price relative to KAMA
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        # RSI filters: avoid overextended conditions
        rsi_not_overbought = rsi[i] < 70
        rsi_not_oversold = rsi[i] > 30
        
        # Weekly trend filter
        uptrend = close[i] > ema_34_1w_aligned[i]
        downtrend = close[i] < ema_34_1w_aligned[i]
        
        # Entry logic: KAMA direction + RSI filter + weekly trend alignment
        if price_above_kama and rsi_not_overbought and uptrend and position != 1:
            position = 1
            signals[i] = 0.25
        elif price_below_kama and rsi_not_oversold and downtrend and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: opposite KAMA signal
        elif position == 1 and price_below_kama:
            position = 0
            signals[i] = 0.0
        elif position == -1 and price_above_kama:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals