#!/usr/bin/env python3
"""
1d_1w_KAMA_Trend_Filter_RSI
Hypothesis: On daily timeframe, use KAMA to filter trend direction and RSI(14) for mean-reversion entries.
Long when price > KAMA and RSI < 30, short when price < KAMA and RSI > 70. Exit when RSI crosses 50.
Uses 1-week trend filter to avoid counter-trend trades in strong trends. Designed for low trade frequency
(10-20/year) by requiring trend alignment and extreme RSI readings. Works in bull/bear via weekly trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_KAMA_Trend_Filter_RSI"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === WEEKLY TREND FILTER ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Weekly EMA(20) for trend
    ema_20 = np.zeros_like(close_1w)
    ema_sum = 0.0
    ema_count = 0
    alpha = 2.0 / (20 + 1)
    for i in range(len(close_1w)):
        if i == 0:
            ema_20[i] = close_1w[i]
        else:
            ema_20[i] = alpha * close_1w[i] + (1 - alpha) * ema_20[i-1]
    
    # Align weekly trend to daily
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20)
    
    # === DAILY KAMA (ER=10) ===
    # Efficiency Ratio
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0)  # placeholder, will compute correctly below
    # Correct volatility calculation: sum of absolute changes over ER period
    er_period = 10
    volatility_sum = np.zeros(n)
    for i in range(n):
        if i < er_period:
            volatility_sum[i] = np.nan
        else:
            volatility_sum[i] = np.sum(np.abs(np.diff(close[i-er_period+1:i+1])))
    # Avoid division by zero
    er = np.where(volatility_sum != 0, change / volatility_sum, 0)
    # Smoothing constants
    sc = (er * (2.0/(2+1) - 2.0/(30+1)) + 2.0/(30+1)) ** 2  # fast=2, slow=30
    # KAMA
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === DAILY RSI(14) ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    rsi = np.zeros(n)
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    for i in range(n):
        if i < 14:
            avg_gain[i] = np.nan
            avg_loss[i] = np.nan
            rsi[i] = 50.0  # neutral
        elif i == 14:
            avg_gain[i] = np.mean(gain[1:15])
            avg_loss[i] = np.mean(loss[1:15])
            if avg_loss[i] == 0:
                rsi[i] = 100.0
            else:
                rsi[i] = 100 - (100 / (1 + avg_gain[i] / avg_loss[i]))
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
            if avg_loss[i] == 0:
                rsi[i] = 100.0
            else:
                rsi[i] = 100 - (100 / (1 + avg_gain[i] / avg_loss[i]))
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # start after warmup
        # Skip if indicators not available
        if (np.isnan(trend_1w_aligned[i]) or np.isnan(kama[i]) or np.isnan(rsi[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend alignment: price should be on same side of weekly EMA
        bullish_trend = close[i] > trend_1w_aligned[i]
        bearish_trend = close[i] < trend_1w_aligned[i]
        
        # Entry conditions
        long_setup = bullish_trend and (close[i] > kama[i]) and (rsi[i] < 30)
        short_setup = bearish_trend and (close[i] < kama[i]) and (rsi[i] > 70)
        
        # Exit conditions: RSI crosses 50
        exit_long = rsi[i] > 50
        exit_short = rsi[i] < 50
        
        if long_setup and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_setup and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals