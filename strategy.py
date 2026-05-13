#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dTrend_Volume
Hypothesis: Camarilla pivot levels (R3/S3) act as strong support/resistance in 4h timeframe.
Breakout above R3 or below S3 with volume confirmation and 1d trend filter captures
institutional breakout moves. Works in both bull and bear markets by following
the dominant 1d trend. Volume filter ensures breakout validity. Target: 20-40 trades/year.
"""

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
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
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla pivot levels for previous day
    # Typical price = (high + low + close) / 3
    typical_price = (high + low + close) / 3.0
    # Daily high, low, close (using 24h lookback for 4h data)
    # We'll use rolling window of 6 for 4h (6*4h = 24h)
    if len(close) < 6:
        return np.zeros(n)
    
    # Calculate daily OHLC from 4h data (6 periods = 1 day)
    daily_high = pd.Series(high).rolling(window=6, min_periods=6).max().shift(6).values
    daily_low = pd.Series(low).rolling(window=6, min_periods=6).min().shift(6).values
    daily_close = pd.Series(close).rolling(window=6, min_periods=6).last().shift(6).values
    
    # Camarilla levels
    R3 = daily_close + (daily_high - daily_low) * 1.1 / 4
    S3 = daily_close - (daily_high - daily_low) * 1.1 / 4
    
    # 1d trend filter: EMA34 on daily close
    # Need to get 1d data for proper trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    uptrend_1d = df_1d['close'].values > ema_34_1d
    downtrend_1d = df_1d['close'].values < ema_34_1d
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d)
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        if np.isnan(R3[i]) or np.isnan(S3[i]) or np.isnan(uptrend_1d_aligned[i]) or np.isnan(downtrend_1d_aligned[i]):
            signals[i] = 0.0
            continue
            
        if position == 0:
            # LONG: breakout above R3 with volume and 1d uptrend
            if close[i] > R3[i] and volume_ok[i] and uptrend_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: breakdown below S3 with volume and 1d downtrend
            elif close[i] < S3[i] and volume_ok[i] and downtrend_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price closes below R3 or trend changes
            if close[i] < R3[i] or not uptrend_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price closes above S3 or trend changes
            if close[i] > S3[i] or not downtrend_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals