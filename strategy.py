#!/usr/bin/env python3
# 1d_1w_KAMA_RSI_Trend_Signal
# Hypothesis: Use weekly KAMA to determine trend direction (bullish if price > KAMA, bearish if price < KAMA). Enter long/short on daily RSI extremes (RSI < 30 or > 70) only when aligned with weekly trend. Exit when RSI returns to neutral range (40-60). Designed for low-frequency, high-conviction trades with minimal churn.

name = "1d_1w_KAMA_RSI_Trend_Signal"
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
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for KAMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate weekly KAMA (ER = 10)
    def kama(close, er_period=10):
        n = len(close)
        kama_arr = np.full(n, np.nan)
        if n == 0:
            return kama_arr
        kama_arr[0] = close[0]
        for i in range(1, n):
            change = abs(close[i] - close[i-1])
            volatility = np.sum(np.abs(np.diff(close[max(0, i-er_period+1):i+1])))
            er = change / volatility if volatility != 0 else 0
            sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # smoothing constant
            kama_arr[i] = kama_arr[i-1] + sc * (close[i] - kama_arr[i-1])
        return kama_arr
    
    kama_1w = kama(close_1w, er_period=10)
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    # Calculate daily RSI (14)
    def rsi(close, period=14):
        n = len(close)
        rsi_arr = np.full(n, np.nan)
        if n < period:
            return rsi_arr
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros(n)
        avg_loss = np.zeros(n)
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        for i in range(period+1, n):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
        rsi_arr[period:] = 100 - (100 / (1 + rs[period:]))
        return rsi_arr
    
    rsi_14 = rsi(close)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(kama_1w_aligned[i]) or np.isnan(rsi_14[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price above weekly KAMA (bullish trend) AND RSI oversold (< 30)
            if close[i] > kama_1w_aligned[i] and rsi_14[i] < 30:
                signals[i] = 0.25
                position = 1
            # Enter short: price below weekly KAMA (bearish trend) AND RSI overbought (> 70)
            elif close[i] < kama_1w_aligned[i] and rsi_14[i] > 70:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI returns to neutral (>= 40) or trend turns bearish
            if rsi_14[i] >= 40 or close[i] < kama_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI returns to neutral (<= 60) or trend turns bullish
            if rsi_14[i] <= 60 or close[i] > kama_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals