#!/usr/bin/env python3
"""
1h RSI Extreme Reversion with 4h Trend Filter
Hypothesis: RSI extremes on 1h combined with 4h trend direction provide high-probability mean reversion entries.
In bull markets: buy RSI<30 in 4h uptrend. In bear markets: sell RSI>70 in 4h downtrend.
Uses 4h trend for direction, 1h for precise entry timing. Targets 15-30 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_rsi(close, period=14):
    """Calculate Relative Strength Index"""
    if len(close) < period + 1:
        return np.full_like(close, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    avg_gain[0] = gain[0]
    avg_loss[0] = loss[0]
    
    for i in range(1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h EMA34 for trend
    close_4h = df_4h['close'].values
    ema_4h = np.zeros_like(close_4h)
    ema_4h[0] = close_4h[0]
    for i in range(1, len(close_4h)):
        ema_4h[i] = (close_4h[i] * 2 + ema_4h[i-1] * 33) / 35  # EMA34 approx
    
    # Align 4h EMA to 1h
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Calculate 1h RSI
    rsi_1h = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Warmup
    
    for i in range(start_idx, n):
        if np.isnan(rsi_1h[i]) or np.isnan(ema_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend: price above/below EMA34
        uptrend_4h = close[i] > ema_4h_aligned[i]
        downtrend_4h = close[i] < ema_4h_aligned[i]
        
        if position == 0:
            # Long: RSI oversold in 4h uptrend
            if rsi_1h[i] < 30 and uptrend_4h:
                signals[i] = 0.20
                position = 1
            # Short: RSI overbought in 4h downtrend
            elif rsi_1h[i] > 70 and downtrend_4h:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: RSI overbought or trend change
            if rsi_1h[i] > 70 or not uptrend_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: RSI oversold or trend change
            if rsi_1h[i] < 30 or not downtrend_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_RSI_Extreme_Reversion_4hTrendFilter"
timeframe = "1h"
leverage = 1.0