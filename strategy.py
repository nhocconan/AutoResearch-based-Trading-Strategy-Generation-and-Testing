#!/usr/bin/env python3
# 1d_KAMA_With_1wTrend_Filter
# Hypothesis: KAMA (Kaufman Adaptive Moving Average) trend following on daily timeframe with weekly trend filter.
# Uses price efficiency ratio to adapt smoothing - fast in trends, slow in ranging markets.
# Weekly trend filter ensures we only trade in direction of higher timeframe trend.
# Target: 15-25 trades/year to minimize fee drag on 1d timeframe.

name = "1d_KAMA_With_1wTrend_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_kama(close, er_length=10, fast_ema=2, slow_ema=30):
    """Calculate Kaufman Adaptive Moving Average"""
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0) if len(close) == 1 else None
    
    if volatility is None:
        # Calculate volatility as sum of absolute changes over er_length period
        volatility = np.full_like(change, np.nan)
        for i in range(len(change)):
            if i < er_length:
                volatility[i] = np.nan
            else:
                volatility[i] = np.sum(np.abs(np.diff(close[i-er_length:i+1])))
    
    # Avoid division by zero
    er = np.where(volatility > 0, change / volatility, 0)
    # For first er_length periods, set ER to 0
    er[:er_length] = 0
    
    # Smoothing constants
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
    
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[0] = close[0]
    
    for i in range(1, len(close)):
        if np.isnan(sc[i]) or np.isnan(kama[i-1]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
            
    return kama

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Weekly trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1w_up = close_1w > ema50_1w
    trend_1w_down = close_1w < ema50_1w
    
    # Align weekly trend to daily
    trend_1w_up_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_up.astype(float))
    trend_1w_down_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_down.astype(float))
    
    # Calculate KAMA on daily data
    kama = calculate_kama(close, er_length=10, fast_ema=2, slow_ema=30)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(trend_1w_up_aligned[i]) or np.isnan(trend_1w_down_aligned[i]) or
            np.isnan(kama[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price crosses above KAMA with weekly uptrend
            if (close[i] > kama[i] and close[i-1] <= kama[i-1] and
                trend_1w_up_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short: price crosses below KAMA with weekly downtrend
            elif (close[i] < kama[i] and close[i-1] >= kama[i-1] and
                  trend_1w_down_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price crosses below KAMA or weekly trend turns down
            if (close[i] < kama[i] and close[i-1] >= kama[i-1] or
                trend_1w_up_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price crosses above KAMA or weekly trend turns up
            if (close[i] > kama[i] and close[i-1] <= kama[i-1] or
                trend_1w_down_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals