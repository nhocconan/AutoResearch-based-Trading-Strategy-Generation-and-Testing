#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_Volume
Hypothesis: Daily Camarilla pivot levels (R1/S1) act as key support/resistance on 12h chart.
Price tends to breakout from R1/S1 with volume confirmation when aligned with daily trend.
In bear markets, breaks below S1 with volume continuation signal short opportunities.
In bull markets, breaks above R1 with volume continuation signal long opportunities.
Designed for low trade frequency (12-37/year) to work in both bull and bear markets by
trading institutional levels with trend and volume filters.
"""

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels"""
    range_val = high - low
    pivot = (high + low + close) / 3.0
    r1 = close + range_val * 1.1 / 12
    r2 = close + range_val * 1.1 / 6
    r3 = close + range_val * 1.1 / 4
    r4 = close + range_val * 1.1 / 2
    s1 = close - range_val * 1.1 / 12
    s2 = close - range_val * 1.1 / 6
    s3 = close - range_val * 1.1 / 4
    s4 = close - range_val * 1.1 / 2
    return pivot, r1, r2, r3, r4, s1, s2, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla levels and trend
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily Camarilla levels
    camarilla = calculate_camarilla(
        df_1d['high'].values,
        df_1d['low'].values,
        df_1d['close'].values
    )
    _, r1, r2, r3, r4, s1, s2, s3, s4 = camarilla
    
    # Align Camarilla levels to 12h timeframe
    r1_12h = align_htf_to_ltf(prices, df_1d, r1)
    s1_12h = align_htf_to_ltf(prices, df_1d, s1)
    
    # Daily EMA34 trend filter
    ema_34 = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume confirmation: > 1.8x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_confirm = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        if position == 0:
            # BREAKOUT LONG: Price breaks above R1 with volume confirmation and daily uptrend
            if close[i] > r1_12h[i] and close[i-1] <= r1_12h[i-1] and volume_confirm[i] and close[i] > ema_34_12h[i]:
                signals[i] = 0.25
                position = 1
            # BREAKOUT SHORT: Price breaks below S1 with volume confirmation and daily downtrend
            elif close[i] < s1_12h[i] and close[i-1] >= s1_12h[i-1] and volume_confirm[i] and close[i] < ema_34_12h[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S1 (stop) or reaches R2 (take profit)
            if close[i] < s1_12h[i] or close[i] >= r2_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R1 (stop) or reaches S2 (take profit)
            if close[i] > r1_12h[i] or close[i] <= s2_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals