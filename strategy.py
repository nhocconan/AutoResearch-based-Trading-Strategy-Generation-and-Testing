#!/usr/bin/env python3
"""
12h_KAMA_Trend_With_1wEMA34_TrendFilter
Hypothesis: Kaufman Adaptive Moving Average (KAMA) on 12h captures trend direction, filtered by 1-week EMA34 to ensure alignment with higher timeframe trend. Works in bull markets (trend following) and bear markets (avoids false signals during sideways chop). Uses volume confirmation to filter weak breakouts. Target: 15-30 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_kama(close, period=10, fast=2, slow=30):
    """Calculate Kaufman Adaptive Moving Average"""
    change = np.abs(np.diff(close, n=period))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
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
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # 1w EMA34 for trend filter
    close_1w = pd.Series(df_1w['close'].values)
    ema34_1w = close_1w.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate KAMA on 12h data
    kama = calculate_kama(close, period=10, fast=2, slow=30)
    
    # Volume spike detection (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup period
    start_idx = 30  # need 30 for KAMA and EMA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above KAMA with volume spike and uptrend on 1w
            if (close[i] > kama[i] and volume_spike[i] and 
                close[i] > ema34_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA with volume spike and downtrend on 1w
            elif (close[i] < kama[i] and volume_spike[i] and 
                  close[i] < ema34_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price returns below KAMA or trend fails
            if (close[i] < kama[i] or 
                close[i] < ema34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns above KAMA or trend fails
            if (close[i] > kama[i] or 
                close[i] > ema34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_KAMA_Trend_With_1wEMA34_TrendFilter"
timeframe = "12h"
leverage = 1.0