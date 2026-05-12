#!/usr/bin/env python3
# 1d_1w_KAMA_Trend_With_Volume_Confirmation
# Hypothesis: Uses 1d KAMA to determine primary trend direction and 1w for higher timeframe trend filter.
# Enters long when price crosses above KAMA with weekly uptrend and volume confirmation.
# Enters short when price crosses below KAMA with weekly downtrend and volume confirmation.
# Uses volume spike (>1.5x 20-period average) for confirmation to reduce false signals.
# Designed for low trade frequency (~30-100 total trades over 4 years) to minimize fee drift.
# Works in bull/bear markets by following weekly trend while using 1d KAMA for precise entries.

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
    
    # Volume spike: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Daily KAMA for trend
    # KAMA parameters: ER = 10, Fast = 2, Slow = 30
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close))
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Weekly EMA20 for trend filter
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align all indicators to daily timeframe
    kama_aligned = kama  # already on daily
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        if (np.isnan(ema_20_1w_aligned[i]) or
            np.isnan(kama_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price crosses above KAMA + weekly EMA20 uptrend + volume spike
            if (close[i] > kama_aligned[i] and close[i-1] <= kama_aligned[i-1] and
                close[i] > ema_20_1w_aligned[i] and
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price crosses below KAMA + weekly EMA20 downtrend + volume spike
            elif (close[i] < kama_aligned[i] and close[i-1] >= kama_aligned[i-1] and
                  close[i] < ema_20_1w_aligned[i] and
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below KAMA OR closes below weekly EMA20
            if (close[i] < kama_aligned[i] and close[i-1] >= kama_aligned[i-1]) or \
               (close[i] < ema_20_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above KAMA OR closes above weekly EMA20
            if (close[i] > kama_aligned[i] and close[i-1] <= kama_aligned[i-1]) or \
               (close[i] > ema_20_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals