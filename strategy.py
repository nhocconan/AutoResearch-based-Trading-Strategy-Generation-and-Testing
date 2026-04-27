#!/usr/bin/env python3
"""
1d_KAMA_Trend_With_RSI_Momentum
Hypothesis: KAMA adapts to market efficiency, reducing noise in choppy conditions. Combined with RSI momentum (14-period) and price above/below KAMA, it captures trend persistence. Weekly trend filter (EMA50) ensures alignment with higher-timeframe momentum. Designed for low trade frequency (~10-20 trades/year) to minimize fee drag on 1d timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA parameters
    er_period = 10
    fast_ema = 2
    slow_ema = 30
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(close, n=er_period))  # |close[t] - close[t-er_period]|
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # sum of |close[i] - close[i-1]| over er_period
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    
    # Smoothing constants
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI (14-period)
    rsi_period = 14
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    avg_gain[rsi_period] = np.mean(gain[:rsi_period])
    avg_loss[rsi_period] = np.mean(loss[:rsi_period])
    
    for i in range(rsi_period + 1, n):
        avg_gain[i] = (avg_gain[i-1] * (rsi_period - 1) + gain[i-1]) / rsi_period
        avg_loss[i] = (avg_loss[i-1] * (rsi_period - 1) + loss[i-1]) / rsi_period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for KAMA, RSI, and weekly EMA
    start_idx = max(er_period, rsi_period, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(ema50_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        kama_val = kama[i]
        rsi_val = rsi[i]
        weekly_trend = ema50_1w_aligned[i]
        
        if position == 0:
            # Long: price above KAMA, RSI > 50 (bullish momentum), and weekly uptrend
            if close[i] > kama_val and rsi_val > 50 and close[i] > weekly_trend:
                signals[i] = size
                position = 1
            # Short: price below KAMA, RSI < 50 (bearish momentum), and weekly downtrend
            elif close[i] < kama_val and rsi_val < 50 and close[i] < weekly_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below KAMA or RSI turns bearish (<40) or weekly trend turns down
            if close[i] < kama_val or rsi_val < 40 or close[i] < weekly_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above KAMA or RSI turns bullish (>60) or weekly trend turns up
            if close[i] > kama_val or rsi_val > 60 or close[i] > weekly_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_KAMA_Trend_With_RSI_Momentum"
timeframe = "1d"
leverage = 1.0