#!/usr/bin/env python3
"""
4h_RelativeStrengthIndex_With_DailyTrend_Filter
Hypothesis: RSI(14) mean-reversion on 4h timeframe, filtered by daily EMA50 trend direction.
In bull markets (price > daily EMA50), we take long signals when RSI < 30.
In bear markets (price < daily EMA50), we take short signals when RSI > 70.
This avoids counter-trend trades and focuses on mean-reversion within the dominant trend.
Designed for low frequency (~15-30 trades/year) with strong performance in both bull and bear markets.
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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily EMA50 trend filter
    close_1d = df_1d['close'].values
    ema50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema50_1d[49] = np.mean(close_1d[0:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_1d)):
            ema50_1d[i] = close_1d[i] * alpha + ema50_1d[i-1] * (1 - alpha)
    
    # Calculate RSI(14) on 4h close
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    
    # Wilder's smoothing: first average is simple average
    if n >= 14:
        avg_gain[13] = np.mean(gain[1:14])  # gain[1] to gain[13] (13 periods)
        avg_loss[13] = np.mean(loss[1:14])
        for i in range(14, n):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.full(n, np.nan)
    rsi = np.full(n, 50.0)  # default to neutral
    valid = (avg_loss != 0) & ~np.isnan(avg_loss)
    rs[valid] = avg_gain[valid] / avg_loss[valid]
    rsi[valid] = 100 - (100 / (1 + rs[valid]))
    # Handle case where avg_loss is zero (all gains)
    rsi[avg_loss == 0] = 100
    
    # Align daily EMA50 to 4h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 14)
    
    for i in range(start_idx, n):
        if np.isnan(ema50_1d_aligned[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long signal: RSI oversold in uptrend (price above daily EMA50)
            if close[i] > ema50_1d_aligned[i] and rsi[i] < 30:
                signals[i] = 0.25
                position = 1
            # Short signal: RSI overbought in downtrend (price below daily EMA50)
            elif close[i] < ema50_1d_aligned[i] and rsi[i] > 70:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI overbought or trend turns down
            if rsi[i] > 70 or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI oversold or trend turns up
            if rsi[i] < 30 or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_RelativeStrengthIndex_With_DailyTrend_Filter"
timeframe = "4h"
leverage = 1.0