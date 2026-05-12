#!/usr/bin/env python3
# 1d_1w_KAMA_Trend_With_Volume_Confirmation
# Hypothesis: Uses KAMA (Kaufman Adaptive Moving Average) on 1d to identify adaptive trend direction.
# Enters long when price crosses above KAMA with volume confirmation and 1w uptrend.
# Enters short when price crosses below KAMA with volume confirmation and 1w downtrend.
# Uses 1w EMA50 as trend filter to avoid counter-trend trades.
# Designed for low trade frequency (~30-100 total trades over 4 years) to minimize fee drag.
# Works in bull/bear markets by following 1w trend while using 1d KAMA cross for precise entries.

name = "1d_1w_KAMA_Trend_With_Volume_Confirmation"
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
    volume = prices['volume'].values
    
    # Volume spike: >1.5x 20-period average (on 1d timeframe)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Daily KAMA (Kaufman Adaptive Moving Average)
    # Efficiency Ratio (ER) = |close - close[10]| / sum(|close - close[1]|) over 10 periods
    # Smoothing Constant (SC) = [ER * (fastest SC - slowest SC) + slowest SC]^2
    # where fastest SC = 2/(2+1), slowest SC = 2/(30+1)
    close_series = pd.Series(close)
    change = abs(close_series - close_series.shift(10))
    volatility = abs(close_series - close_series.shift(1)).rolling(window=10, min_periods=10).sum()
    er = change / volatility
    er = er.fillna(0)  # Handle division by zero
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all indicators to 1d timeframe
    kama_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), kama)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if (np.isnan(kama_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price crosses above KAMA + 1w EMA50 uptrend + volume spike
            if (close[i] > kama_aligned[i] and 
                close[i-1] <= kama_aligned[i-1] and  # crossed above
                close[i] > ema_50_1w_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price crosses below KAMA + 1w EMA50 downtrend + volume spike
            elif (close[i] < kama_aligned[i] and 
                  close[i-1] >= kama_aligned[i-1] and  # crossed below
                  close[i] < ema_50_1w_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below KAMA OR closes below 1w EMA50
            if (close[i] < kama_aligned[i] and close[i-1] >= kama_aligned[i-1]) or \
               (close[i] < ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above KAMA OR closes above 1w EMA50
            if (close[i] > kama_aligned[i] and close[i-1] <= kama_aligned[i-1]) or \
               (close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals