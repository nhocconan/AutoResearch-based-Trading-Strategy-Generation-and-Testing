#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h KAMA Trend with 1d RSI Filter and Volume Confirmation
# Uses Kaufman's Adaptive Moving Average (KAMA) on 12h to identify adaptive trend direction.
# Enters long when KAMA turns upward, RSI(14) on 1d is above 50 (bullish momentum), and 1d volume is above average.
# Enters short when KAMA turns downward, RSI(14) on 1d is below 50 (bearish momentum), and 1d volume is above average.
# Exits when KAMA direction reverses or volume drops below average.
# Designed to work in both bull and bear markets by adapting to volatility and using volume/momentum confirmation.
# Target: 20-50 trades per symbol over 4 years (5-12.5/year)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h and 1d data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 12h KAMA (adaptive moving average)
    close_12h = df_12h['close'].values
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close_12h, n=10))  # |close[t] - close[t-10]|
    volatility = np.sum(np.abs(np.diff(close_12h, n=1)), axis=0)  # sum of |close[t] - close[t-1]| over 10 periods
    # Handle first 10 values
    change = np.concatenate([[np.nan] * 10, change])
    volatility = np.concatenate([[np.nan] * 10, volatility])
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2 / (2 + 1) - 2 / (30 + 1)) + 2 / (30 + 1)) ** 2  # fast=2, slow=30
    # Calculate KAMA
    kama = np.full_like(close_12h, np.nan)
    kama[0] = close_12h[0]
    for i in range(1, len(close_12h)):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close_12h[i] - kama[i-1])
    
    # Calculate 1d RSI (14-period)
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    # First average gain/loss
    avg_gain = np.concatenate([[np.nan] * 14, [np.mean(gain[:14])]])
    avg_loss = np.concatenate([[np.nan] * 14, [np.mean(loss[:14])]])
    # Smooth subsequent values
    for i in range(15, len(gain)+1):
        avg_gain = np.append(avg_gain, (avg_gain[-1] * 13 + gain[i-1]) / 14)
        avg_loss = np.append(avg_loss, (avg_loss[-1] * 13 + loss[i-1]) / 14)
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 12h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_12h, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50  # for KAMA and RSI calculations
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get previous KAMA to detect direction change
        prev_kama = kama_aligned[i-1] if i > 0 else kama_aligned[i]
        price = close[i]
        vol_1d_current = vol_1d[i] if i < len(vol_1d) else vol_1d[-1]
        
        if position == 0:
            # Long setup: KAMA turning up, RSI > 50, volume above average
            if (kama_aligned[i] > prev_kama and 
                rsi_aligned[i] > 50 and 
                vol_1d_current > vol_ma_1d_aligned[i]):
                position = 1
                signals[i] = position_size
            # Short setup: KAMA turning down, RSI < 50, volume above average
            elif (kama_aligned[i] < prev_kama and 
                  rsi_aligned[i] < 50 and 
                  vol_1d_current > vol_ma_1d_aligned[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: KAMA turns down or volume drops below average
            if kama_aligned[i] < prev_kama or vol_1d_current < vol_ma_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: KAMA turns up or volume drops below average
            if kama_aligned[i] > prev_kama or vol_1d_current < vol_ma_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_KAMA_1dRSI_Volume"
timeframe = "12h"
leverage = 1.0