#!/usr/bin/env python3
"""
12h_KAMA_Direction_RSI14_Trend_Filter
Hypothesis: Uses 12h KAMA to determine trend direction (bullish/bearish), RSI14 on 12h for overbought/oversold confirmation, and 1d volume spike for entry timing. Enters long when KAMA turns up, RSI14 < 40, and volume spike; enters short when KAMA turns down, RSI14 > 60, and volume spike. Exits on opposite KAMA crossover. Designed to work in both bull and bear markets by adapting to trend via KAMA and avoiding extremes via RSI filter. Targets 15-25 trades/year via strict entry conditions.
"""

name = "12h_KAMA_Direction_RSI14_Trend_Filter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_kama(close, er_length=10, fast_sc=2, slow_sc=30):
    """Calculate Kaufman Adaptive Moving Average"""
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    # Use rolling sum for volatility
    volatility_rolling = pd.Series(volatility).rolling(window=er_length, min_periods=er_length).sum().values
    # Avoid division by zero
    er = np.where(volatility_rolling != 0, change / volatility_rolling, 0)
    sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    return kama

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # 12h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 12h KAMA Trend ---
    kama = calculate_kama(close, er_length=10, fast_sc=2, slow_sc=30)
    kama_prev = np.roll(kama, 1)
    kama_prev[0] = kama[0]
    kama_up = kama > kama_prev
    kama_down = kama < kama_prev
    
    # --- 12h RSI14 ---
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # --- 1d Volume Spike ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = vol_1d / vol_ma_1d
    vol_ratio_1d = np.nan_to_num(vol_ratio_1d, nan=1.0)
    vol_ratio_12h = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_ratio_12h[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation threshold
        volume_spike = vol_ratio_12h[i] > 1.5
        
        if position == 0:
            # Long: KAMA turning up + RSI14 < 40 (not overbought) + volume spike
            if (kama_up[i] and 
                rsi[i] < 40 and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: KAMA turning down + RSI14 > 60 (not oversold) + volume spike
            elif (kama_down[i] and 
                  rsi[i] > 60 and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit on opposite KAMA crossover
            if position == 1:
                # Exit long: KAMA turns down
                if kama_down[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: KAMA turns up
                if kama_up[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals