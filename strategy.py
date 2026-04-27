#!/usr/bin/env python3
"""
Hypothesis: 1-day RSI(2) extreme reversion with 1-week trend filter.
Enters long when RSI(2) < 5 and price > weekly EMA(50) (oversold in uptrend).
Enters short when RSI(2) > 95 and price < weekly EMA(50) (overbought in downtrend).
Uses RSI(2) for extreme mean reversion signals and weekly EMA for trend alignment.
Designed to work in both bull and bear markets by taking counter-trend entries only
when aligned with the higher-timeframe trend. Target: 10-20 trades/year per symbol
to minimize fee drag and avoid overtrading.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get daily data for RSI(2)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate daily RSI(2)
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Use Wilder's smoothing (alpha = 1/period)
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[0] = gain[0]
    avg_loss[0] = loss[0]
    
    for i in range(1, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * (2-1) + gain[i]) / 2
        avg_loss[i] = (avg_loss[i-1] * (2-1) + loss[i]) / 2
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate weekly EMA(50)
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align indicators to lower timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need RSI(2) and weekly EMA(50)
    start_idx = max(2, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(rsi_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current daily price
        price_now = close[i]
        rsi_now = rsi_aligned[i]
        trend_1w = ema_50_1w_aligned[i]
        
        # Entry conditions
        if position == 0:
            # Long: RSI(2) < 5 (extremely oversold) + price > weekly EMA(50) (uptrend)
            if rsi_now < 5 and price_now > trend_1w:
                signals[i] = size
                position = 1
            # Short: RSI(2) > 95 (extremely overbought) + price < weekly EMA(50) (downtrend)
            elif rsi_now > 95 and price_now < trend_1w:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI(2) > 60 (normal) or price < weekly EMA(50) (trend change)
            if rsi_now > 60 or price_now < trend_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: RSI(2) < 40 (normal) or price > weekly EMA(50) (trend change)
            if rsi_now < 40 or price_now > trend_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_RSI2_Extreme_1wTrend"
timeframe = "1d"
leverage = 1.0