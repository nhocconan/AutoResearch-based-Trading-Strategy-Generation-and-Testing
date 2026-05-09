#!/usr/bin/env python3
# Hypothesis: 4h KAMA trend with 12h RSI filter and volume confirmation
# Long when KAMA is rising and RSI < 60 (pullback in uptrend)
# Short when KAMA is falling and RSI > 40 (bounce in downtrend)
# Uses adaptive trend following with momentum filter to avoid whipsaws
# Target: 80-120 total trades over 4 years (20-30/year) with size 0.25

name = "4h_KAMA_RSI_Volume_Trend"
timeframe = "4h"
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
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    # ER = Efficiency Ratio, SC = Smoothing Constant
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0)  # placeholder, will compute properly
    # Recalculate volatility as sum of absolute changes over ER period
    er_period = 10
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    
    # Vectorized ER calculation
    abs_diff = np.abs(np.diff(close, prepend=close[0]))
    change_abs = np.abs(np.diff(close, prepend=close[0]))
    
    # Initialize arrays
    er = np.zeros(n)
    sc = np.zeros(n)
    kama = np.zeros(n)
    
    # Calculate Efficiency Ratio
    for i in range(er_period, n):
        direction = np.abs(close[i] - close[i - er_period])
        volatility_sum = np.sum(np.abs(np.diff(close[i - er_period:i + 1], prepend=close[i - er_period])))
        if volatility_sum > 0:
            er[i] = direction / volatility_sum
        else:
            er[i] = 0
    
    # Calculate Smoothing Constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate 12h RSI for momentum filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    
    # RSI calculation
    delta = np.diff(df_12h['close'].values, prepend=df_12h['close'].values[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(len(gain))
    avg_loss = np.zeros(len(loss))
    
    # Wilder's smoothing
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.where(np.isnan(rsi), 50, rsi)  # handle division by zero
    
    # Align RSI to 4h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_12h, rsi)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Need enough data for KAMA and volume
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama[i]) or np.isnan(rsi_aligned[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: KAMA rising, RSI < 60 (not overbought), volume confirmation
            if (kama[i] > kama[i-1] and 
                rsi_aligned[i] < 60 and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: KAMA falling, RSI > 40 (not oversold), volume confirmation
            elif (kama[i] < kama[i-1] and 
                  rsi_aligned[i] > 40 and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: KAMA falling or RSI > 70 (overbought)
            if (kama[i] < kama[i-1]) or (rsi_aligned[i] > 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: KAMA rising or RSI < 30 (oversold)
            if (kama[i] > kama[i-1]) or (rsi_aligned[i] < 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals