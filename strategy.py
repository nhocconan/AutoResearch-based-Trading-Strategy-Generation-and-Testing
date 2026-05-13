#!/usr/bin/env python3
"""
6h_Camarilla_R3S3_Breakout_1dTrend_Volume
Hypothesis: Camarilla pivot breakouts at R3/S3 levels with 1-day trend filter and volume confirmation capture strong directional moves while avoiding false breakouts in ranging markets. Designed for low trade frequency (15-30/year) with clear entry/exit rules that work in both bull and bear markets by requiring trend alignment and volume validation.
"""

name = "6h_Camarilla_R3S3_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels from previous day
    def calculate_camarilla(h, l, c):
        range_val = h - l
        if range_val == 0:
            return c, c, c, c, c, c, c, c
        r4 = c + (range_val * 1.1 / 2)
        r3 = c + (range_val * 1.1/4)
        r2 = c + (range_val * 1.1/6)
        r1 = c + (range_val * 1.1/12)
        s1 = c - (range_val * 1.1/12)
        s2 = c - (range_val * 1.1/6)
        s3 = c - (range_val * 1.1/4)
        s4 = c - (range_val * 1.1/2)
        return r1, r2, r3, r4, s1, s2, s3, s4
    
    # Get previous day's OHLC for Camarilla calculation
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    # Calculate Camarilla levels for each bar
    r1 = np.zeros(n)
    r2 = np.zeros(n)
    r3 = np.zeros(n)
    r4 = np.zeros(n)
    s1 = np.zeros(n)
    s2 = np.zeros(n)
    s3 = np.zeros(n)
    s4 = np.zeros(n)
    
    for i in range(n):
        r1[i], r2[i], r3[i], r4[i], s1[i], s2[i], s3[i], s4[i] = calculate_camarilla(
            prev_high[i], prev_low[i], prev_close[i]
        )
    
    # Get 1-day trend filter (EMA 34)
    df_1d = get_htf_data(prices, '1d')
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        if position == 0:
            # LONG: Price breaks above R3 with 1-day uptrend and volume confirmation
            if close[i] > r3[i] and close[i] > ema_34_1d_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 with 1-day downtrend and volume confirmation
            elif close[i] < s3[i] and close[i] < ema_34_1d_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below R2 or 1-day trend turns down
            if close[i] < r2[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above S2 or 1-day trend turns up
            if close[i] > s2[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals