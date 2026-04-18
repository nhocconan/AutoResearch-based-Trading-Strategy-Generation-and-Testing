#!/usr/bin/env python3
"""
1d_Weekly_Trend_Follower
Hypothesis: Uses 1-week SMA trend filter on daily data with daily RSI pullback entries.
Long when weekly SMA rising and daily RSI < 30; short when weekly SMA falling and daily RSI > 70.
Targets 10-20 trades/year by combining weekly trend filter with daily oversold/overbought entries.
Works in bull markets via trend following and in bear markets via mean-reversion within trend.
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
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    weekly_close = df_weekly['close'].values
    
    # Calculate weekly SMA(10) for trend filter
    weekly_sma10 = np.full(len(weekly_close), np.nan)
    if len(weekly_close) >= 10:
        weekly_sma10[9] = np.mean(weekly_close[0:10])
        for i in range(10, len(weekly_close)):
            weekly_sma10[i] = (weekly_sma10[i-1] * 9 + weekly_close[i]) / 10
    
    # Align weekly SMA to daily timeframe
    weekly_sma10_aligned = align_htf_to_ltf(prices, df_weekly, weekly_sma10)
    
    # Calculate daily RSI(14)
    rsi = np.full(n, np.nan)
    if n >= 14:
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full(n, np.nan)
        avg_loss = np.full(n, np.nan)
        
        avg_gain[13] = np.mean(gain[0:14])
        avg_loss[13] = np.mean(loss[0:14])
        
        for i in range(14, n):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
        rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(14, 10)
    
    for i in range(start_idx, n):
        if (np.isnan(weekly_sma10_aligned[i]) or np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        # Weekly trend: rising if current > previous, falling if current < previous
        if i > 0 and not np.isnan(weekly_sma10_aligned[i-1]):
            weekly_rising = weekly_sma10_aligned[i] > weekly_sma10_aligned[i-1]
            weekly_falling = weekly_sma10_aligned[i] < weekly_sma10_aligned[i-1]
        else:
            weekly_rising = False
            weekly_falling = False
        
        if position == 0:
            # Long: weekly uptrend + daily RSI oversold (<30)
            if weekly_rising and rsi[i] < 30:
                signals[i] = 0.25
                position = 1
            # Short: weekly downtrend + daily RSI overbought (>70)
            elif weekly_falling and rsi[i] > 70:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: weekly trend turns down OR RSI overbought (>70)
            if not weekly_rising or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: weekly trend turns up OR RSI oversold (<30)
            if not weekly_falling or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Weekly_Trend_Follower"
timeframe = "1d"
leverage = 1.0