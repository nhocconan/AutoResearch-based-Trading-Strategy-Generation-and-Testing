#!/usr/bin/env python3
# 4h_KAMA_Direction_RSI_Filter
# Hypothesis: Use Kaufman Adaptive Moving Average (KAMA) to determine trend direction,
# combined with RSI for momentum confirmation and volume filter for entry confirmation.
# KAMA adapts to market noise, making it effective in both trending and ranging markets.
# RSI filters out weak momentum, and volume filter ensures institutional participation.
# Target: 20-30 trades/year to minimize fee drag while capturing significant moves.
# Designed to work in both bull and bear markets by following adaptive trend.

name = "4h_KAMA_Direction_RSI_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data for trend context (optional, can use 4h only)
    # But per instructions, we'll use 4h for primary signals with optional 1d trend filter
    # However, to keep it simple and focused, we'll use 4h-based KAMA and RSI
    # If we want 1d trend filter, we uncomment below:
    # df_1d = get_htf_data(prices, '1d')
    # ... but let's keep it 4h self-contained to avoid overcomplication
    
    # Calculate KAMA on 4h close
    close = prices['close'].values
    volume = prices['volume'].values
    
    # KAMA parameters
    er_period = 10      # Efficiency Ratio period
    fast_sc = 2         # Fast EMA constant
    slow_sc = 30        # Slow EMA constant
    
    # Calculate Change and Volatility
    change = np.abs(np.diff(close, prepend=close[0]))  # |close - close_prev|
    volatility = np.zeros(n)
    for i in range(er_period, n):
        volatility[i] = np.sum(np.abs(np.diff(close[i-er_period+1:i+1])))  # sum of abs changes
    
    # Avoid division by zero
    volatility = np.where(volatility == 0, 1, volatility)
    
    # Efficiency Ratio
    er = np.zeros(n)
    er[er_period:] = change[er_period:] / volatility[er_period:]
    
    # Smoothing Constants
    sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI calculation (14-period)
    rsi_period = 14
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # First average gain/loss
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[rsi_period] = np.mean(gain[1:rsi_period+1])
    avg_loss[rsi_period] = np.mean(loss[1:rsi_period+1])
    
    # Wilder's smoothing
    for i in range(rsi_period+1, n):
        avg_gain[i] = (avg_gain[i-1] * (rsi_period-1) + gain[i]) / rsi_period
        avg_loss[i] = (avg_loss[i-1] * (rsi_period-1) + loss[i]) / rsi_period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)  # avoid div by zero
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: current volume > 1.5x 20-period EMA volume
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need KAMA (er_period), RSI (rsi_period), and volume EMA
    start_idx = max(er_period, rsi_period, 20) + 5  # extra buffer
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or
            np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above KAMA, RSI > 50 (bullish momentum), and volume confirmation
            if close[i] > kama[i] and rsi[i] > 50 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA, RSI < 50 (bearish momentum), and volume confirmation
            elif close[i] < kama[i] and rsi[i] < 50 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price below KAMA or RSI < 40 (losing momentum)
            if close[i] < kama[i] or rsi[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price above KAMA or RSI > 60 (losing momentum)
            if close[i] > kama[i] or rsi[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals