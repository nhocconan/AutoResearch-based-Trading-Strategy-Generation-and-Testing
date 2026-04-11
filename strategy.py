#!/usr/bin/env python3
"""
1d_1w_kama_rsi_chop_v1
Strategy: Daily KAMA direction with RSI momentum filter and Choppiness regime filter
Timeframe: 1d
Leverage: 1.0
Hypothesis: Uses KAMA to capture trend direction, RSI for momentum confirmation, and Choppiness index to avoid ranging markets. Weekly trend filter ensures alignment with higher timeframe momentum. Designed for 30-100 trades over 4 years with focus on avoiding false signals in chop. Works in both bull and bear markets by adapting to regime conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_kama_rsi_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # KAMA (Kaufman Adaptive Moving Average) - 14 period
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=1))
    change = np.insert(change, 0, 0)  # align length
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0)  # placeholder, will compute properly below
    
    # Proper ER calculation
    er = np.zeros(n)
    for i in range(10, n):  # min_periods=10 for ER
        if i >= 10:
            direction = np.abs(close[i] - close[i-9])
            volatility = np.sum(np.abs(np.diff(close[i-9:i+1])))+1e-10
            er[i] = direction / volatility if volatility > 0 else 0
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # for EMA(2)
    slow_sc = 2 / (30 + 1)  # for EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI (14-period)
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    for i in range(14, n):
        if i == 14:
            avg_gain[i] = np.mean(gain[i-13:i+1])
            avg_loss[i] = np.mean(loss[i-13:i+1])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index (14-period)
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR (14-period)
    atr = np.zeros(n)
    for i in range(14, n):
        if i == 14:
            atr[i] = np.mean(tr[i-13:i+1])
        else:
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Sum of TR over 14 periods
    sum_tr = np.zeros(n)
    for i in range(14, n):
        sum_tr[i] = np.sum(tr[i-13:i+1])
    
    # Highest high and lowest low over 14 periods
    max_high = np.zeros(n)
    min_low = np.zeros(n)
    for i in range(14, n):
        max_high[i] = np.max(high[i-13:i+1])
        min_low[i] = np.min(low[i-13:i+1])
    
    # Choppiness Index
    chop = np.zeros(n)
    for i in range(14, n):
        if sum_tr[i] > 0 and max_high[i] > min_low[i]:
            chop[i] = 100 * np.log10(sum_tr[i] / (max_high[i] - min_low[i])) / np.log10(14)
        else:
            chop[i] = 50  # neutral
    
    # Weekly trend filter - EMA20 on weekly close
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # warmup period
        # Skip if any required data is invalid
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(ema_20_1w_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        
        # KAMA direction
        kama_up = price_close > kama[i]
        kama_down = price_close < kama[i]
        
        # RSI momentum filter
        rsi_momentum_up = rsi[i] > 55
        rsi_momentum_down = rsi[i] < 45
        
        # Choppiness regime filter - avoid extreme chop
        not_choppy = chop[i] < 61.8  # below chop threshold = trending
        
        # Weekly trend alignment
        weekly_uptrend = price_close > ema_20_1w_aligned[i]
        weekly_downtrend = price_close < ema_20_1w_aligned[i]
        
        # Long conditions: price above KAMA + RSI momentum + not choppy + weekly uptrend
        long_signal = kama_up and rsi_momentum_up and not_choppy and weekly_uptrend
        
        # Short conditions: price below KAMA + RSI momentum + not choppy + weekly downtrend
        short_signal = kama_down and rsi_momentum_down and not_choppy and weekly_downtrend
        
        # Exit conditions: opposite KAMA cross or RSI reversal
        exit_long = position == 1 and (price_close < kama[i] or rsi[i] < 40)
        exit_short = position == -1 and (price_close > kama[i] or rsi[i] > 60)
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals