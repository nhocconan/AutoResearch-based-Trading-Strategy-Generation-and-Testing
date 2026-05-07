#!/usr/bin/env python3
# 4h_KAMA_1dTrend_Volume
# Hypothesis: KAMA adapts to market efficiency, filtering noise in chop and capturing trends. 
# Long when price > KAMA and above 1d EMA34 trend, short when price < KAMA and below 1d EMA34.
# Volume confirmation ensures momentum behind moves. Works in bull (trend following) and bear (mean reversion via KAMA adaptation).
# Target: 20-40 trades/year to stay within optimal frequency range and minimize fee drag.

name = "4h_KAMA_1dTrend_Volume"
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
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_4h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate KAMA (4h timeframe)
    # Efficiency Ratio = |close - close[10]| / sum(|close - close[1]|) over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close[t] - close[t-10]|
    volatility = np.abs(np.diff(close, n=1))  # |close[t] - close[t-1]|
    
    # Pad arrays for alignment
    change_padded = np.concatenate([np.full(10, np.nan), change])
    volatility_padded = np.concatenate([np.full(1, np.nan), volatility])
    
    # Sum volatility over 10 periods
    vol_sum = np.convolve(volatility_padded, np.ones(10), mode='valid')
    # Avoid division by zero
    er = np.where(vol_sum > 0, change_padded[10:] / vol_sum, 0)
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # Start at index 9 to have enough data
    
    for i in range(10, n):
        if np.isnan(kama[i-1]) or np.isnan(sc[i]):
            kama[i] = close[i]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Volume spike confirmation (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):  # Start after EMA warmup
        # Skip if any critical value is NaN
        if (np.isnan(kama[i]) or np.isnan(ema_34_1d_4h[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price > KAMA and above 1d EMA34 + volume spike
            if close[i] > kama[i] and close[i] > ema_34_1d_4h[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price < KAMA and below 1d EMA34 + volume spike
            elif close[i] < kama[i] and close[i] < ema_34_1d_4h[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price crosses below KAMA or below 1d EMA34
            if close[i] < kama[i] or close[i] < ema_34_1d_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price crosses above KAMA or above 1d EMA34
            if close[i] > kama[i] or close[i] > ema_34_1d_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals