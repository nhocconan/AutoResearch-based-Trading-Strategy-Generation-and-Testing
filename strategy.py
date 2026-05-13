#!/usr/bin/env python3
"""
1d_Weekly_Pivot_R3_S3_Breakout_Trend_Volume
Hypothesis: Weekly Camarilla pivot levels (R3/S3) act as strong support/resistance on daily timeframe.
Breakout above weekly R3 with daily EMA trend and volume confirmation signals long.
Breakdown below weekly S3 with daily EMA trend and volume confirmation signals short.
Uses daily EMA50 trend filter and volume > 1.5x average to reduce false signals.
Target: 15-25 trades/year per symbol to avoid fee drift and work in both bull/bear markets.
"""

name = "1d_Weekly_Pivot_R3_S3_Breakout_Trend_Volume"
timeframe = "1d"
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
    
    # Daily EMA50 for trend filter
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend = close > ema_50
    downtrend = close < ema_50
    
    # Volume confirmation: > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    # Get weekly OHLC for Camarilla pivot calculation
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 1:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous weekly bar
    # R3 = close + 1.1*(high - low)
    # S3 = close - 1.1*(high - low)
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    r3 = weekly_close + 1.1 * (weekly_high - weekly_low)
    s3 = weekly_close - 1.1 * (weekly_high - weekly_low)
    
    # Align weekly levels to daily timeframe (already delayed by weekly close)
    r3_aligned = align_htf_to_ltf(prices, df_weekly, r3)
    s3_aligned = align_htf_to_ltf(prices, df_weekly, s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if position == 0:
            # LONG: break above weekly R3, uptrend, volume confirmation
            if close[i] > r3_aligned[i] and uptrend[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: break below weekly S3, downtrend, volume confirmation
            elif close[i] < s3_aligned[i] and downtrend[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price falls back below weekly R3 or trend reverses
            if close[i] < r3_aligned[i] or not uptrend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price rises back above weekly S3 or trend reverses
            if close[i] > s3_aligned[i] or not downtrend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals