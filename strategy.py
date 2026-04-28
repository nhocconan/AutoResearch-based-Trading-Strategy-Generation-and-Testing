#!/usr/bin/env python3
"""
4h_KAMA_Trend_RSI_Filter_VolumeSpike
Hypothesis: Uses KAMA (Kaufman Adaptive Moving Average) to capture trend direction,
filtered by RSI(14) for momentum and volume spike for confirmation. KAMA adapts
to market noise, reducing whipsaw in ranging markets while capturing trends.
Volume spike ensures participation. Designed for 4h timeframe to balance trade
frequency and signal quality. Targets 20-40 trades/year.
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
    
    # KAMA parameters
    er_len = 10
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    
    # Calculate Efficiency Ratio
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0) if False else None
    # Manual calculation for efficiency
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.zeros_like(close)
    for i in range(1, len(close)):
        volatility[i] = volatility[i-1] + np.abs(close[i] - close[i-1])
        if i >= er_len:
            volatility[i] -= np.abs(close[i-er_len] - close[i-er_len-1])
    
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume spike (2x 20-period MA)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma_20)
    
    # Signals
    kama_up = kama > np.roll(kama, 1)
    kama_down = kama < np.roll(kama, 1)
    rsi_long = rsi > 50
    rsi_short = rsi < 50
    
    # Entry conditions
    long_entry = kama_up & rsi_long & vol_spike
    short_entry = kama_down & rsi_short & vol_spike
    
    # Exit conditions (opposite signal)
    long_exit = kama_down
    short_exit = kama_up
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14, 20)  # max of lookback periods
    
    for i in range(start_idx, n):
        if long_entry[i] and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry[i] and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit[i] and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit[i] and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_KAMA_Trend_RSI_Filter_VolumeSpike"
timeframe = "4h"
leverage = 1.0