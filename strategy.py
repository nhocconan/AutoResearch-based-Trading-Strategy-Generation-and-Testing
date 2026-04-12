#!/usr/bin/env python3
"""
4h_1d_KAMA_RSI_v1
Hypothesis: On 4h timeframe, use KAMA trend direction filtered by RSI(14) > 50 for longs and < 50 for shorts.
Add 1d volume confirmation: require current volume > 1.5x 20-period average volume on 1d timeframe.
KAMA adapts to market noise, reducing whipsaws in sideways markets. RSI ensures momentum alignment.
Volume filter ensures participation from higher timeframe activity.
Works in bull via KAMA uptrend + RSI > 50, works in bear via KAMA downtrend + RSI < 50.
Target: 20-40 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_KAMA_RSI_v1"
timeframe = "4h"
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
    
    # 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # KAMA calculation on 4h close
    def calculate_kama(close_prices, length=10, fast=2, slow=30):
        # Efficiency Ratio
        change = np.abs(np.diff(close_prices, prepend=close_prices[0]))
        volatility = np.abs(np.diff(close_prices))
        er = np.zeros_like(close_prices, dtype=np.float64)
        for i in range(1, len(close_prices)):
            if volatility[i] != 0:
                er[i] = change[i] / volatility[i]
            else:
                er[i] = 0
        # Smoothing constants
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama = np.zeros_like(close_prices)
        kama[0] = close_prices[0]
        for i in range(1, len(close_prices)):
            kama[i] = kama[i-1] + sc[i] * (close_prices[i] - kama[i-1])
        return kama
    
    kama = calculate_kama(close, length=10, fast=2, slow=30)
    
    # RSI(14) on 4h close
    def calculate_rsi(close_prices, length=14):
        delta = np.diff(close_prices, prepend=close_prices[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(gain)
        avg_loss = np.zeros_like(loss)
        for i in range(1, len(gain)):
            if i < length:
                avg_gain[i] = (avg_gain[i-1] * (i-1) + gain[i]) / i
                avg_loss[i] = (avg_loss[i-1] * (i-1) + loss[i]) / i
            else:
                avg_gain[i] = (avg_gain[i-1] * (length-1) + gain[i]) / length
                avg_loss[i] = (avg_loss[i-1] * (length-1) + loss[i]) / length
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = calculate_rsi(close, length=14)
    
    # 1d volume average (20-period)
    vol_20 = np.zeros_like(df_1d['volume'].values)
    for i in range(len(df_1d)):
        if i < 20:
            vol_20[i] = np.mean(df_1d['volume'].values[:i+1]) if i > 0 else df_1d['volume'].values[i]
        else:
            vol_20[i] = np.mean(df_1d['volume'].values[i-19:i+1])
    vol_20_avg = vol_20
    vol_20_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_20_avg)
    
    # Align KAMA and RSI to 4h (they are already on 4h, but ensure no look-ahead)
    kama_aligned = kama  # already on 4h
    rsi_aligned = rsi    # already on 4h
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # warmup for KAMA/RSI
        # Skip if any data invalid
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(vol_20_avg_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume filter: current volume > 1.5x 20-period average volume on 1d
        vol_filter = volume[i] > 1.5 * vol_20_avg_aligned[i]
        
        # Entry conditions
        long_entry = kama[i] > close[i] and rsi[i] > 50 and vol_filter
        short_entry = kama[i] < close[i] and rsi[i] < 50 and vol_filter
        
        # Exit conditions: reverse signal
        long_exit = kama[i] < close[i] or rsi[i] <= 50
        short_exit = kama[i] > close[i] or rsi[i] >= 50
        
        # Signal logic
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals