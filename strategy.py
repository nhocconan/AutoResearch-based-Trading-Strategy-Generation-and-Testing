#!/usr/bin/env python3
# 1D_KAMA_Trend_With_Volume_Confirmation_v2
# Hypothesis: On 1d timeframe, use Kaufman Adaptive Moving Average (KAMA) for trend direction with volume confirmation to filter false signals. 
# Enter long when price crosses above KAMA with volume > 1.5x average, short when price crosses below KAMA with volume confirmation.
# Uses weekly trend filter to avoid counter-trend trades in strong trends. Designed for lower trade frequency (~10-25/year) to minimize fee drag.
# Works in both bull and bear markets by adapting to price efficiency and using volume as confirmation.

name = "1D_KAMA_Trend_With_Volume_Confirmation_v2"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_kama(close, er_length=10, fast_sc=2, slow_sc=30):
    """Calculate Kaufman Adaptive Moving Average"""
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0) if len(change) > 1 else np.zeros_like(change)
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    # For multi-period efficiency ratio
    change_t = np.abs(np.diff(close, n=er_length, prepend=close[:er_length]))
    volatility_t = np.array([np.sum(np.abs(np.diff(close[i:i+er_length+1]))) if i+er_length < len(close) else 0 
                            for i in range(len(close))])
    er_t = np.where(volatility_t != 0, change_t / volatility_t, 0)
    sc = (er_t * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    return kama

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate KAMA on daily data
    kama = calculate_kama(close, er_length=10, fast_sc=2, slow_sc=30)
    
    # Weekly trend: price above/below 20-period EMA
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    weekly_uptrend = close_1w > ema_20_1w
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_avg * 1.5)
    
    # Align weekly trend to daily
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data for indicators
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(kama[i]) or np.isnan(weekly_uptrend_aligned[i]) or np.isnan(volume_confirm[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price crosses above KAMA + weekly uptrend + volume confirmation
            if close[i] > kama[i] and close[i-1] <= kama[i-1] and weekly_uptrend_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price crosses below KAMA + weekly downtrend + volume confirmation
            elif close[i] < kama[i] and close[i-1] >= kama[i-1] and not weekly_uptrend_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below KAMA or weekly trend turns down
            if close[i] < kama[i] or not weekly_uptrend_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above KAMA or weekly trend turns up
            if close[i] > kama[i] or weekly_uptrend_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals