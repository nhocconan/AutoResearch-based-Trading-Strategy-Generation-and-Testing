#!/usr/bin/env python3
# 1D_KAMA_Reversal_With_Volume_Filter
# Hypothesis: Uses Kaufman Adaptive Moving Average (KAMA) to detect trend reversals, combined with volume spikes and RSI confirmation for high-probability entries. Designed to work in both bull and bear markets by avoiding whipsaws through adaptive smoothing and volume confirmation. Target: 10-25 trades/year on 1d timeframe to minimize fee drag while capturing major reversals.

name = "1D_KAMA_Reversal_With_Volume_Filter"
timeframe = "1d"
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
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # Calculate weekly EMA21 for trend filter
    ema_21_1w = pd.Series(df_1w['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Calculate KAMA ( Kaufman Adaptive Moving Average )
    # Parameters: ER length=10, Fast SC=2/(2+1), Slow SC=2/(30+1)
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0) if False else None  # placeholder
    
    # Proper volatility calculation: sum of absolute changes over ER period
    er_period = 10
    change_arr = np.abs(np.diff(close, prepend=close[0]))
    volatility_arr = np.zeros_like(change)
    for i in range(len(change)):
        if i < er_period:
            volatility_arr[i] = np.sum(change_arr[max(0, i-er_period+1):i+1])
        else:
            volatility_arr[i] = np.sum(change_arr[i-er_period+1:i+1])
    
    # Avoid division by zero
    er = np.where(volatility_arr != 0, change_arr / volatility_arr, 0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    
    # Initialize KAMA
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI for confirmation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    for i in range(14, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume spike detection: 1.5x average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 21)  # Ensure we have volume MA, RSI, and EMA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(ema_21_1w_aligned[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price crosses above KAMA with volume spike and RSI > 50, weekly uptrend
            if close[i] > kama[i] and close[i-1] <= kama[i-1] and volume[i] > 1.5 * vol_ma[i] and rsi[i] > 50 and close[i] > ema_21_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below KAMA with volume spike and RSI < 50, weekly downtrend
            elif close[i] < kama[i] and close[i-1] >= kama[i-1] and volume[i] > 1.5 * vol_ma[i] and rsi[i] < 50 and close[i] < ema_21_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price crosses below KAMA or trend failure
            if close[i] < kama[i] or close[i] < ema_21_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price crosses above KAMA or trend failure
            if close[i] > kama[i] or close[i] > ema_21_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals