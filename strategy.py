#!/usr/bin/env python3
"""
1d_KAMA_Trend_Filtered_By_Volume_and_1wTrend
Hypothesis: 1-day Kaufman Adaptive Moving Average (KAMA) with volume confirmation
and weekly trend filter captures medium-term trends while avoiding whipsaws.
KAMA adapts to market noise, reducing false signals in ranging markets.
Volume ensures conviction, and weekly trend alignment improves win rate.
Target: 15-25 trades/year per symbol, suitable for 1d timeframe.
"""

name = "1d_KAMA_Trend_Filtered_By_Volume_and_1wTrend"
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
    volume = prices['volume'].values
    
    # KAMA parameters
    er_length = 10
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1) # EMA(30)
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(close, n=er_length))
    volatility = np.sum(np.abs(np.diff(close)), axis=0) if len(change.shape) > 0 else np.sum(np.abs(np.diff(close)))
    # Handle 1D case properly
    volatility = np.array([np.sum(np.abs(np.diff(close[i:i+er_length]))) for i in range(len(change))])
    er = np.where(volatility != 0, change / volatility, 0)
    
    # Smoothing constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.full_like(close, np.nan, dtype=np.float64)
    kama[er_length] = close[er_length]
    for i in range(er_length + 1, n):
        if not np.isnan(sc[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Volume confirmation: >1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * vol_ma)
    
    # Weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after KAMA warmup
        if (np.isnan(kama[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(volume_confirmed[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price above KAMA + weekly uptrend + volume confirmation
            if (close[i] > kama[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                volume_confirmed[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price below KAMA + weekly downtrend + volume confirmation
            elif (close[i] < kama[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume_confirmed[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price below KAMA or weekly downtrend
            if (close[i] < kama[i]) or \
               (close[i] < ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price above KAMA or weekly uptrend
            if (close[i] > kama[i]) or \
               (close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals