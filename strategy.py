#!/usr/bin/env python3
"""
1h_4h_Camarilla_R3_S3_Breakout_1dTrend
Hypothesis: Use daily close above/below 200 EMA as trend filter. In uptrend (price > EMA200), go long when price breaks above Camarilla R3 level from prior 4h bar. In downtrend (price < EMA200), go short when price breaks below Camarilla S3 level from prior 4h bar. Camarilla levels provide intraday support/resistance based on prior bar's range. This strategy limits entries by requiring trend alignment and breakouts, reducing trade frequency. Designed for 1h timeframe with 4h levels for structure and daily trend filter to work in both bull (catch breakouts) and bear (catch breakdowns) markets.
"""

name = "1h_4h_Camarilla_R3_S3_Breakout_1dTrend"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 4h data for Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each 4h bar: R3, S3
    # Camarilla: R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
    hl_range = df_4h['high'].values - df_4h['low'].values
    camarilla_r3 = df_4h['close'].values + 1.1 * hl_range / 2.0
    camarilla_s3 = df_4h['close'].values - 1.1 * hl_range / 2.0
    
    # Align Camarilla levels to 1h timeframe (wait for 4h bar close)
    r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3)
    
    # Get daily data for 200 EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Calculate daily EMA200
    ema_200_1d = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_200_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price > EMA200 (uptrend) and price breaks above R3 from prior 4h bar
            if close[i] > ema_200_aligned[i] and high[i] > r3_aligned[i]:
                signals[i] = 0.20
                position = 1
            # SHORT: price < EMA200 (downtrend) and price breaks below S3 from prior 4h bar
            elif close[i] < ema_200_aligned[i] and low[i] < s3_aligned[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price breaks below S3 or trend reverses
            if low[i] < s3_aligned[i] or close[i] < ema_200_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: price breaks above R3 or trend reverses
            if high[i] > r3_aligned[i] or close[i] > ema_200_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals