#!/usr/bin/env python3
"""
1d_KAMA_Trend_With_WeeklyTrend_Filter
Hypothesis: Daily KAMA direction + weekly trend filter + volume confirmation.
KAMA adapts to market noise, reducing whipsaw in sideways markets.
Weekly trend filter ensures alignment with higher timeframe momentum.
Volume confirmation filters out low-conviction moves.
Designed for 1d timeframe to target 15-25 trades/year, minimizing fee drag.
Works in both bull (follows trend) and bear (avoids counter-trend trades) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # KAMA (Kaufman Adaptive Moving Average) parameters
    er_len = 10
    fast_ma = 2
    slow_ma = 30
    
    # Calculate Efficiency Ratio
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close, prepend=close[0]))
    er = np.zeros(n)
    for i in range(er_len, n):
        price_change = np.abs(close[i] - close[i-er_len])
        sum_volatility = np.sum(volatility[i-er_len+1:i+1])
        if sum_volatility > 0:
            er[i] = price_change / sum_volatility
        else:
            er[i] = 0
    
    # Smoothing constants
    sc = (er * (2/(fast_ma+1) - 2/(slow_ma+1)) + 2/(slow_ma+1)) ** 2
    
    # Calculate KAMA
    kama = np.full(n, np.nan)
    kama[0] = close[0]
    for i in range(1, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Weekly trend filter: EMA34 on weekly close
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation: >1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(35, 20)  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(kama[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        kama_val = kama[i]
        weekly_ema = ema_34_1w_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price above KAMA, above weekly EMA, with volume spike
            if price > kama_val and price > weekly_ema and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA, below weekly EMA, with volume spike
            elif price < kama_val and price < weekly_ema and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price crosses below KAMA OR weekly trend turns down
            if price < kama_val:
                signals[i] = 0.0
                position = 0
            elif price < weekly_ema:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price crosses above KAMA OR weekly trend turns up
            if price > kama_val:
                signals[i] = 0.0
                position = 0
            elif price > weekly_ema:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_KAMA_Trend_With_WeeklyTrend_Filter"
timeframe = "1d"
leverage = 1.0