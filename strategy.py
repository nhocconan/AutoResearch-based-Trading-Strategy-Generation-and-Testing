# 1d_Weekly_KAMA_RSI_Trend_Follow
# Hypothesis: Use weekly KAMA direction (trend) with daily RSI for entries.
# In weekly bullish regime (KAMA rising): go long when daily RSI crosses above 50.
# In weekly bearish regime (KAMA falling): go short when daily RSI crosses below 50.
# Weekly timeframe ensures fewer trades (7-25/year), reducing fee drag.
# RSI provides timely entries within the trend. Works in both bull and bear markets.
# Uses daily timeframe for signal generation with weekly trend filter.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_Weekly_KAMA_RSI_Trend_Follow"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Get weekly data for KAMA trend
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    def kama(close, period=10, fast=2, slow=30):
        # Efficiency Ratio
        change = np.abs(np.diff(close, n=period))
        abs_change = np.abs(np.diff(close, n=1))
        er = np.zeros_like(close)
        er[period:] = change[period-1:] / np.where(np.sum(abs_change.reshape(-1, period), axis=1) == 0, 1, np.sum(abs_change.reshape(-1, period), axis=1))
        # Smoothing constants
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        # KAMA calculation
        kama_vals = np.zeros_like(close)
        kama_vals[period] = close[period]
        for i in range(period+1, len(close)):
            kama_vals[i] = kama_vals[i-1] + sc[i] * (close[i] - kama_vals[i-1])
        return kama_vals
    
    kama_1w = kama(close_1w, period=10, fast=2, slow=30)
    # Weekly trend: rising KAMA = bullish, falling KAMA = bearish
    kama_rising = np.diff(kama_1w, prepend=kama_1w[0]) > 0
    
    # Get daily data for RSI
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate RSI (14-period)
    def rsi(close, period=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        # Wilder's smoothing
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        avg_gain[period] = np.mean(gain[1:period+1])
        avg_loss[period] = np.mean(loss[1:period+1])
        for i in range(period+1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        rs = np.where(avg_loss == 0, 100, avg_gain / avg_loss)
        rsi_vals = 100 - (100 / (1 + rs))
        return rsi_vals
    
    rsi_1d = rsi(close_1d, period=14)
    
    # Align weekly KAMA trend to daily
    kama_rising_aligned = align_htf_to_ltf(prices, df_1w, kama_rising.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 15  # Ensure RSI and KAMA are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(kama_rising_aligned[i]) or np.isnan(rsi_1d[i]):
            signals[i] = 0.0
            continue
        
        kama_bullish = kama_rising_aligned[i] > 0.5
        rsi_val = rsi_1d[i]
        
        # Entry logic
        if position == 0:
            if kama_bullish and rsi_val > 50:
                # Weekly bullish + RSI > 50 → long
                signals[i] = 0.25
                position = 1
            elif not kama_bullish and rsi_val < 50:
                # Weekly bearish + RSI < 50 → short
                signals[i] = -0.25
                position = -1
        
        # Exit logic: opposite RSI cross
        elif position == 1:
            if rsi_val < 50:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            if rsi_val > 50:
                signals[i] = 0.0
                position = 0
    
    return signals