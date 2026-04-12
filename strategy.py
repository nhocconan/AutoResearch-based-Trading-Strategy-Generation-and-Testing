#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h_1d_kama_rsi_v1
# Uses KAMA direction as primary trend filter, RSI(14) for mean-reversion entries,
# and volume confirmation. In bull markets, buy dips when RSI<30 with rising KAMA.
# In bear markets, sell rallies when RSI>70 with falling KAMA.
# Volume > 1.5x 20-period average confirms institutional participation.
# Target: 40-80 trades/year per symbol for balanced frequency and edge.

name = "4h_1d_kama_rsi_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for KAMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate KAMA on daily closes
    close_1d = df_1d['close'].values
    kama = calculate_kama(close_1d, 10, 2, 30)
    # Align KAMA to 4h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # KAMA direction: 1=rising, -1=falling, 0=flat
    kama_dir = np.where(kama_aligned > np.roll(kama_aligned, 1), 1, 
                        np.where(kama_aligned < np.roll(kama_aligned, 1), -1, 0))
    kama_dir[0] = 0  # first value
    
    # RSI(14) on 4h closes
    rsi = calculate_rsi(close, 14)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # start after warmup
        # Skip if values not ready
        if np.isnan(kama_aligned[i]) or np.isnan(rsi[i]) or np.isnan(vol_confirm[i]):
            signals[i] = 0.0
            continue
        
        # Check volume filter
        if not vol_confirm[i]:
            # Hold current position if volume filter fails
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Long signal: RSI < 30 (oversold) AND KAMA rising (bullish bias)
        if rsi[i] < 30 and kama_dir[i] == 1 and position != 1:
            position = 1
            signals[i] = 0.25
        # Short signal: RSI > 70 (overbought) AND KAMA falling (bearish bias)
        elif rsi[i] > 70 and kama_dir[i] == -1 and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: opposite RSI extreme
        elif rsi[i] > 70 and position == 1:
            position = 0
            signals[i] = 0.0
        elif rsi[i] < 30 and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

def calculate_rsi(prices, period=14):
    """Calculate RSI with proper Wilder's smoothing"""
    delta = np.diff(prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.full_like(prices, np.nan, dtype=float)
    avg_loss = np.full_like(prices, np.nan, dtype=float)
    
    if len(gain) < period:
        return avg_gain  # all NaN
    
    # First values: simple average
    avg_gain[period] = np.mean(gain[:period])
    avg_loss[period] = np.mean(loss[:period])
    
    # Subsequent values: Wilder's smoothing
    for i in range(period + 1, len(prices)):
        avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
    
    # Avoid division by zero
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_kama(close, fast=2, slow=30, er_period=10):
    """Calculate Kaufman Adaptive Moving Average"""
    # Efficiency Ratio
    change = np.abs(np.diff(close, er_period))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # needs fixing
    
    # Correct volatility calculation
    volatility = np.full_like(change, np.nan, dtype=float)
    for i in range(len(change)):
        volatility[i] = np.sum(np.abs(np.diff(close[i:i+er_period+1])))
    
    er = np.where(volatility != 0, change / volatility, 0)
    
    # Smoothing constants
    sc = np.square(er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1))
    
    # KAMA calculation
    kama = np.full_like(close, np.nan, dtype=float)
    kama[er_period] = close[er_period]  # start with close
    
    for i in range(er_period + 1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama