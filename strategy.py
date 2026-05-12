#!/usr/bin/env python3
# 1d_KAMA_Trend_RSI_Filter
# Hypothesis: On 1d timeframe, use Kaufman Adaptive Moving Average (KAMA) for trend direction.
# Enter long when price crosses above KAMA with RSI < 50 and weekly EMA50 uptrend.
# Enter short when price crosses below KAMA with RSI > 50 and weekly EMA50 downtrend.
# Exit when price crosses back over KAMA.
# Uses weekly trend filter to avoid counter-trend trades, targeting 10-25 trades/year for low friction.
# Works in bull via KAMA uptrend entries and in bear via KAMA downtrend entries.

name = "1d_KAMA_Trend_RSI_Filter"
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
    volume = prices['volume'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    def calculate_kama(close, period=10, fast=2, slow=30):
        change = np.abs(np.diff(close, n=period))
        volatility = np.sum(np.abs(np.diff(close)), axis=0)
        er = np.where(volatility != 0, change / volatility, 0)
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama = np.full_like(close, np.nan, dtype=float)
        kama[period] = close[period]
        for i in range(period+1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    # Calculate RSI
    def calculate_rsi(close, period=14):
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.full_like(close, np.nan, dtype=float)
        avg_loss = np.full_like(close, np.nan, dtype=float)
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        for i in range(period+1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # Calculate KAMA and RSI
    kama = calculate_kama(close, period=10, fast=2, slow=30)
    rsi = calculate_rsi(close, period=14)
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    weekly_close = df_1w['close'].values
    weekly_ema50 = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_ema50_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(weekly_ema50_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        kama_val = kama[i]
        rsi_val = rsi[i]
        weekly_trend = weekly_ema50_aligned[i]
        
        if position == 0:
            # LONG: Price crosses above KAMA with RSI < 50 and weekly uptrend
            if close[i] > kama_val and close[i-1] <= kama[i-1] and rsi_val < 50 and close[i] > weekly_trend:
                signals[i] = 0.25
                position = 1
            # SHORT: Price crosses below KAMA with RSI > 50 and weekly downtrend
            elif close[i] < kama_val and close[i-1] >= kama[i-1] and rsi_val > 50 and close[i] < weekly_trend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses back below KAMA
            if close[i] < kama_val and close[i-1] >= kama[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses back above KAMA
            if close[i] > kama_val and close[i-1] <= kama[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals