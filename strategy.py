#!/usr/bin/env python3
# Hypothesis: 1-day KAMA direction with 1-week EMA50 trend filter and volume confirmation
# Long when KAMA trending upward, price above KAMA, EMA50 uptrend, and volume > 1.5x average
# Short when KAMA trending downward, price below KAMA, EMA50 downtrend, and volume > 1.5x average
# Exit when price crosses below/above KAMA or trend reverses
# Uses KAMA for adaptive trend, weekly EMA for higher timeframe confirmation, volume for conviction
# Designed to capture trending moves in both bull and bear markets with low frequency
# Target: 30-80 total trades over 4 years (7-20/year) with size 0.25

name = "1d_KAMA_Direction_1wEMA50_VolumeConfirmation"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d KAMA (adaptive moving average)
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close[i] - close[i-10]|
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # sum of |close[i] - close[i-1]| over 10 periods
    # Avoid division by zero
    er = np.divide(change, volatility, out=np.zeros_like(change, dtype=float), where=volatility!=0)
    # Smoothing constants
    sc = (er * (2/(2+2) - 2/(30+2)) + 2/(30+2)) ** 2  # fast=2, slow=30
    # Initialize KAMA
    kama = np.full_like(close, np.nan, dtype=float)
    kama[9] = close[9]  # Start after first 10 periods
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i-10] * (close[i-1] - kama[i-1])
    
    # Calculate 1-week EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume confirmation: current volume > 1.5x 50-period average
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean()
    vol_confirm = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Need enough data for KAMA and EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama[i]) or np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price above KAMA, KAMA upward, EMA50 uptrend, volume spike
            if (close[i] > kama[i] and 
                kama[i] > kama[i-1] and 
                ema50_1w_aligned[i] > ema50_1w_aligned[i-1] and
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price below KAMA, KAMA downward, EMA50 downtrend, volume spike
            elif (close[i] < kama[i] and 
                  kama[i] < kama[i-1] and 
                  ema50_1w_aligned[i] < ema50_1w_aligned[i-1] and
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below KAMA or trend reverses
            if (close[i] < kama[i]) or (kama[i] < kama[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above KAMA or trend reverses
            if (close[i] > kama[i]) or (kama[i] > kama[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals