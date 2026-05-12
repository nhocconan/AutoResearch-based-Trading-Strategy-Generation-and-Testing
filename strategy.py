#!/usr/bin/env python3
"""
1d_KAMA_Trend_With_RSI_Filter
Hypothesis: KAMA adapts to market efficiency, filtering noise in chop and catching trends. 
Combined with RSI(2) for mean-reversion entries in the direction of KAMA trend on daily timeframe.
Designed for low trade frequency (10-25/year) to minimize fee drag, works in both bull and bear regimes.
"""

name = "1d_KAMA_Trend_With_RSI_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_kama(close, er_length=10, fast=2, slow=30):
    """Calculate Kaufman Adaptive Moving Average"""
    change = np.abs(np.diff(close, n=er_length))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)
    
    # Handle edge cases
    er = np.zeros_like(close)
    er[er_length:] = change[er_length-1:] / np.maximum(volatility[er_length-1:], 1e-10)
    
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    
    kama = np.zeros_like(close)
    kama[:] = np.nan
    kama[er_length] = close[er_length]
    
    for i in range(er_length + 1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_rsi(close, length=2):
    """Calculate RSI with Wilder's smoothing"""
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    
    # Wilder's smoothing
    avg_gain[length] = np.mean(gain[:length])
    avg_loss[length] = np.mean(loss[:length])
    
    for i in range(length + 1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (length-1) + gain[i-1]) / length
        avg_loss[i] = (avg_loss[i-1] * (length-1) + loss[i-1]) / length
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # Pad beginning
    rsi_full = np.full_like(close, np.nan)
    rsi_full[length:] = rsi[length:]
    
    return rsi_full

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
    if len(df_1w) < 20:
        return np.zeros(n)

    # Calculate KAMA on daily data
    kama = calculate_kama(close, er_length=10, fast=2, slow=30)
    
    # Calculate RSI(2)
    rsi = calculate_rsi(close, length=2)
    
    # Calculate weekly EMA20 for trend filter
    ema20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):  # Start after warmup
        # Get aligned values for current day
        kama_val = kama[i]
        rsi_val = rsi[i]
        ema20_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)[i]
        
        # Skip if any required data is NaN
        if (np.isnan(kama_val) or np.isnan(rsi_val) or 
            np.isnan(ema20_aligned)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price > KAMA (trend up) + RSI(2) < 10 (oversold) + weekly uptrend
            if (close[i] > kama_val and 
                rsi_val < 10 and 
                close[i] > ema20_aligned):
                signals[i] = 0.25
                position = 1
            # SHORT: Price < KAMA (trend down) + RSI(2) > 90 (overbought) + weekly downtrend
            elif (close[i] < kama_val and 
                  rsi_val > 90 and 
                  close[i] < ema20_aligned):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price < KAMA or RSI(2) > 50 (mean reversion complete)
            if (close[i] < kama_val or rsi_val > 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price > KAMA or RSI(2) < 50 (mean reversion complete)
            if (close[i] > kama_val or rsi_val < 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals