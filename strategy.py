#!/usr/bin/env python3
"""
1d_1w_KAMA_Direction_Plus_RSI_With_WeeklyTrend
Hypothesis: On daily timeframe, use KAMA direction for trend bias and RSI for mean-reversion entries, filtered by weekly trend (KAMA slope) to avoid counter-trend trades. KAMA adapts to market noise, reducing whipsaw in sideways markets. RSI(14) < 30 for longs and > 70 for shorts in alignment with weekly trend. Weekly trend filter ensures we only trade in the direction of the higher timeframe momentum. Position size 0.25 to balance opportunity and risk. Target: 20-60 trades over 4 years (5-15/year).
"""

name = "1d_1w_KAMA_Direction_Plus_RSI_With_WeeklyTrend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_kama(close, er_period=10, fast_sc=2, slow_sc=30):
    """Kaufman's Adaptive Moving Average"""
    change = np.abs(close - np.roll(close, er_period))
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0) if False else None  # placeholder for correct calc
    
    # Correct volatility calculation: sum of absolute changes over er_period
    volatility = np.zeros_like(close)
    for i in range(len(close)):
        if i < er_period:
            volatility[i] = np.nan
        else:
            volatility[i] = np.sum(np.abs(np.diff(close[i-er_period:i])))
    
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    sc = np.power(er * (fast_sc - slow_sc) + slow_sc, 2) / 100  # smoothing constant
    
    kama = np.zeros_like(close)
    kama[:] = np.nan
    if len(close) > 0:
        kama[0] = close[0]
        for i in range(1, len(close)):
            if np.isnan(kama[i-1]):
                kama[i] = close[i]
            else:
                kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    return kama

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    
    # Wilder's smoothing
    avg_gain[period] = np.mean(gain[:period])
    avg_loss[period] = np.mean(loss[:period])
    
    for i in range(period + 1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get daily data (same as primary) for KAMA and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate indicators
    kama_1d = calculate_kama(close_1d, 10, 2, 30)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    rsi_1d = calculate_rsi(close_1d, 14)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Weekly KAMA slope for trend filter
    kama_1w = calculate_kama(close_1w, 10, 2, 30)
    # Calculate slope: (current - previous) / previous to get percentage change
    kama_1w_slope = np.zeros_like(kama_1w)
    kama_1w_slope[1:] = (kama_1w[1:] - kama_1w[:-1]) / kama_1w[:-1]
    kama_1w_slope_aligned = align_htf_to_ltf(prices, df_1w, kama_1w_slope)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(kama_1w_slope_aligned[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price > KAMA (uptrend bias) AND RSI < 30 (oversold) AND weekly KAMA slope > 0 (upward momentum)
            if close[i] > kama_1d_aligned[i] and rsi_1d_aligned[i] < 30 and kama_1w_slope_aligned[i] > 0:
                signals[i] = 0.25
                position = 1
            # Short: price < KAMA (downtrend bias) AND RSI > 70 (overbought) AND weekly KAMA slope < 0 (downward momentum)
            elif close[i] < kama_1d_aligned[i] and rsi_1d_aligned[i] > 70 and kama_1w_slope_aligned[i] < 0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price < KAMA (trend bias broken) OR RSI > 70 (overbought)
            if close[i] < kama_1d_aligned[i] or rsi_1d_aligned[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price > KAMA (trend bias broken) OR RSI < 30 (oversold)
            if close[i] > kama_1d_aligned[i] or rsi_1d_aligned[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals