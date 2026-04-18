#!/usr/bin/env python3
"""
4h_KAMA_Direction_Volume_Trend_Filter
Hypothesis: KAMA adapts to market noise, providing reliable trend direction in both bull and bear markets.
Price crossing above/below KAMA with volume confirmation and 12h trend filter captures sustained moves.
Designed for 4h timeframe with tight entry conditions to avoid overtrading (target: 20-50 trades/year).
"""

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
    
    # KAMA (Kaufman Adaptive Moving Average) parameters
    fast_sc = 0.666  # 2/(2+1)
    slow_sc = 0.0645 # 2/(30+1)
    
    # Calculate Efficiency Ratio and Smoothing Constant
    change = np.abs(np.diff(close, n=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # 10-period volatility
    # Handle first 9 values where we don't have 10-period data
    change = np.concatenate([np.full(9, np.nan), change])
    volatility = np.concatenate([np.full(9, np.nan), volatility])
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # Start with first available close
    for i in range(10, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # 12h EMA trend filter
    ema_12h = pd.Series(df_12h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_12h_4h = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume filter: >1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 20  # Warmup for volume MA and KAMA
    
    for i in range(start_idx, n):
        if (np.isnan(kama[i]) or np.isnan(ema_12h_4h[i]) or 
            np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        kama_val = kama[i]
        vol_ok = volume_filter[i]
        ema_trend = ema_12h_4h[i]
        
        if position == 0:
            # Long: price above KAMA with volume in uptrend
            if price > kama_val and vol_ok and price > ema_trend:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA with volume in downtrend
            elif price < kama_val and vol_ok and price < ema_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price crosses below KAMA or trend reverses
            if price < kama_val or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price crosses above KAMA or trend reverses
            if price > kama_val or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_KAMA_Direction_Volume_Trend_Filter"
timeframe = "4h"
leverage = 1.0