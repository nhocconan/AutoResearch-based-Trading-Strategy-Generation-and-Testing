#!/usr/bin/env python3
"""
4h_Stochastic_Trend_Pullback
Hypothesis: Stochastic oscillator (14,3,3) pullbacks in alignment with 4h and 1d trends offer high-probability entries in both bull and bear markets.
Buy when %K crosses above 20 in uptrend, sell when %K crosses below 80 in downtrend, with volume confirmation.
Exit on opposite Stochastic cross or trend reversal. Target: 20-40 trades/year per symbol.
"""

name = "4h_Stochastic_Trend_Pullback"
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
    
    # Stochastic Oscillator (14,3,3)
    k_period = 14
    d_period = 3
    
    # Lowest low and highest high over k_period
    lowest_low = pd.Series(low).rolling(window=k_period, min_periods=k_period).min().values
    highest_high = pd.Series(high).rolling(window=k_period, min_periods=k_period).max().values
    
    # %K
    k_raw = 100 * (close - lowest_low) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    k_raw = np.where(highest_high == lowest_low, 50.0, k_raw)
    k = pd.Series(k_raw).rolling(window=d_period, min_periods=d_period).mean().values
    
    # %D (not used directly but kept for reference)
    d = pd.Series(k).rolling(window=d_period, min_periods=d_period).mean().values
    
    # 4h trend: EMA50
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_4h = close > ema_50
    downtrend_4h = close < ema_50
    
    # 1d trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_1d = df_1d['close'].values > ema_50_1d
    downtrend_1d = df_1d['close'].values < ema_50_1d
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d)
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_conf = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get values
        k_val = k[i]
        k_prev = k[i-1]
        uptrend = uptrend_4h[i]
        downtrend = downtrend_4h[i]
        uptrend_htf = uptrend_1d_aligned[i]
        downtrend_htf = downtrend_1d_aligned[i]
        vol_conf = volume_conf[i]
        
        if position == 0:
            # LONG: %K crosses above 20 from below, uptrend alignment, volume confirmation
            if k_prev <= 20 and k_val > 20 and uptrend and uptrend_htf and vol_conf:
                signals[i] = 0.25
                position = 1
            # SHORT: %K crosses below 80 from above, downtrend alignment, volume confirmation
            elif k_prev >= 80 and k_val < 80 and downtrend and downtrend_htf and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: %K crosses below 80 or 4h trend turns down
            if k_val < 80 and k_prev >= 80 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: %K crosses above 20 or 4h trend turns up
            if k_val > 20 and k_prev <= 20 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals