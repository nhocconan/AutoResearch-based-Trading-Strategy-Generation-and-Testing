#!/usr/bin/env python3
"""
4h_1d_SqueezeBreakout
Hypothesis: Bollinger Band squeeze (low volatility) followed by breakout in direction of 1d KAMA trend.
Works in both bull and bear markets by trading volatility breakouts aligned with higher timeframe trend.
Targets 20-40 trades per year per symbol with strict entry conditions to avoid overtrading.
"""

name = "4h_1d_SqueezeBreakout"
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
    
    # Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2.0
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_std_dev = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper = sma + bb_std * bb_std_dev
    lower = sma - bb_std * bb_std_dev
    bb_width = (upper - lower) / sma  # Normalized width
    
    # Bollinger Band squeeze detection: width < 20th percentile of last 50 periods
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=20).quantile(0.20).values
    squeeze = bb_width < bb_width_percentile
    
    # Breakout detection: close crosses above upper or below lower band
    breakout_up = close > upper
    breakout_down = close < lower
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d trend: KAMA (adaptive moving average)
    # Efficiency Ratio
    change = np.abs(np.diff(close_1d, k=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close_1d, k=1)), axis=0) if len(close_1d) > 1 else np.array([0])
    volatility = pd.Series(volatility).rolling(window=10, min_periods=1).sum().values
    er = np.where(volatility > 0, change / volatility, 0)
    
    # Smoothing constants
    sc = (er * (2/(2+2) - 2/(30+2)) + 2/(30+2)) ** 2  # Fast=2, Slow=30
    
    # KAMA calculation
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    uptrend_1d = close_1d > kama
    downtrend_1d = close_1d < kama
    
    # Align 1d trend to 4h
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d)
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d)
    
    # Entry conditions: squeeze breakout in direction of 1d trend
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if position == 0:
            # LONG: squeeze breakout up + 1d uptrend
            if squeeze[i-1] and breakout_up[i] and uptrend_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: squeeze breakout down + 1d downtrend
            elif squeeze[i-1] and breakout_down[i] and downtrend_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: close below middle band (SMA) or 1d trend turns down
            if close[i] < sma[i] or not uptrend_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: close above middle band (SMA) or 1d trend turns up
            if close[i] > sma[i] or not downtrend_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals