#!/usr/bin/env python3
# 4h_KAMA_Direction_RSI_Trend_Filter
# Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market noise, providing a reliable trend direction.
# In trending markets, price stays above/below KAMA; in ranging markets, frequent crosses are filtered by RSI (40-60).
# Entry: price crosses above KAMA with RSI > 50 (bullish momentum) or below KAMA with RSI < 50 (bearish momentum).
# Exit: price crosses back across KAMA. Uses 1d EMA50 as higher-timeframe trend filter to avoid counter-trend trades.
# Volume confirmation (20-period average) ensures breakouts have conviction. Designed for low trade frequency.

name = "4h_KAMA_Direction_RSI_Trend_Filter"
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
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # KAMA (Kaufman Adaptive Moving Average) - 20 periods
    # ER = Efficiency Ratio = |close - close[10]| / sum(|close - close[1]|) over 10 periods
    # SC = [ER * (fastest - slowest) + slowest]^2
    # KAMA = prev KAMA + SC * (close - prev KAMA)
    # Using common parameters: fast=2, slow=30
    def calculate_kama(price, kama_period=20, fast=2, slow=30):
        n = len(price)
        kama = np.full(n, np.nan)
        if n < kama_period:
            return kama
        
        change = np.abs(np.diff(price, kama_period))  # |close - close[10]|
        volatility = np.sum(np.abs(np.diff(price, 1)), axis=1)  # sum of absolute changes
        # Pad volatility to match change length
        volatility = np.concatenate([np.full(kama_period-1, np.nan), volatility])
        
        er = np.zeros(n)
        er[:] = np.nan
        er[kama_period-1:] = change[kama_period-1:] / volatility[kama_period-1:]
        er[volatility == 0] = 0  # avoid division by zero
        
        sc = (er * (fast - slow) + slow) ** 2
        
        # Initialize KAMA
        kama[kama_period-1] = np.mean(price[:kama_period])
        
        for i in range(kama_period, n):
            if np.isnan(sc[i]):
                kama[i] = kama[i-1]
            else:
                kama[i] = kama[i-1] + sc[i] * (price[i] - kama[i-1])
        
        return kama
    
    kama = calculate_kama(close, 20, 2, 30)
    
    # RSI (14-period)
    def calculate_rsi(price, period=14):
        n = len(price)
        rsi = np.full(n, np.nan)
        if n < period + 1:
            return rsi
        
        delta = np.diff(price)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full(n, np.nan)
        avg_loss = np.full(n, np.nan)
        
        # First average
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        
        # Wilder's smoothing
        for i in range(period+1, n):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = calculate_rsi(close, 14)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation (20-period average)
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p-1, len(arr)):
                res[i] = np.mean(arr[i-p+1:i+1])
        return res
    vol_ma = mean_arr(volume, 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for KAMA, RSI, EMA
    
    for i in range(start_idx, n):
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or \
           np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price crosses above KAMA, RSI > 50, above 1d EMA50, volume confirmation
            if close[i] > kama[i] and close[i-1] <= kama[i-1] and rsi[i] > 50 and close[i] > ema_50_aligned[i] and volume[i] > 1.5 * vol_ma[i]:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below KAMA, RSI < 50, below 1d EMA50, volume confirmation
            elif close[i] < kama[i] and close[i-1] >= kama[i-1] and rsi[i] < 50 and close[i] < ema_50_aligned[i] and volume[i] > 1.5 * vol_ma[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back below KAMA
            if close[i] < kama[i] and close[i-1] >= kama[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back above KAMA
            if close[i] > kama[i] and close[i-1] <= kama[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals