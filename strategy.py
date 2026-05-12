#!/usr/bin/env python3
"""
12h_KAMA_Trend_With_PriceChannel_Exit
Hypothesis: KAMA (Kaufman Adaptive Moving Average) identifies the trend direction on 12h, while price crossing above/below the 12h Donchian(20) channel acts as entry trigger. This combination adapts to both trending and ranging markets, reducing whipsaw. Works in bull/bear by following KAMA trend. Uses 1w trend filter for higher timeframe confirmation to avoid counter-trend trades.
"""

name = "12h_KAMA_Trend_With_PriceChannel_Exit"
timeframe = "12h"
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
    
    # Get 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate KAMA on 12h data
    # Efficiency Ratio (ER) = |close - close[10]| / sum(|close - close[1]|) over 10 periods
    change = np.abs(close - np.roll(close, 10))
    volatility = np.sum(np.abs(np.diff(close, axis=0)), axis=0) if len(close) > 1 else 0
    # Correct volatility calculation for rolling sum
    volatility = np.zeros_like(close)
    for i in range(1, len(close)):
        volatility[i] = volatility[i-1] + np.abs(close[i] - close[i-1])
        if i >= 10:
            volatility[i] -= np.abs(close[i-10] - close[i-11]) if i >= 11 else 0
    # Simpler: use pandas for ER calculation
    close_series = pd.Series(close)
    change = abs(close_series - close_series.shift(10))
    volatility = abs(close_series.diff()).rolling(window=10, min_periods=1).sum()
    er = change / volatility
    er = er.fillna(0).values
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate Donchian channel (20-period) on 12h
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # 1w EMA40 trend filter
    close_1w = df_1w['close'].values
    ema_40_1w = pd.Series(close_1w).ewm(span=40, adjust=False, min_periods=40).mean().values
    ema_40_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_40_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian warmup
        if (np.isnan(kama[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_40_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian high + KAMA uptrend + 1w uptrend
            if (close[i] > donchian_high[i] and 
                close[i] > kama[i] and 
                close[i] > ema_40_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian low + KAMA downtrend + 1w downtrend
            elif (close[i] < donchian_low[i] and 
                  close[i] < kama[i] and 
                  close[i] < ema_40_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below KAMA (trend reversal)
            if close[i] < kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above KAMA (trend reversal)
            if close[i] > kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals