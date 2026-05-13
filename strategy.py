#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1dTrend_Volume
Hypothesis: Daily trend direction filters Camarilla R3/S3 breakouts on 12h for higher probability trades.
Uses volume confirmation to avoid false breakouts. Designed for low trade frequency (12-37/year) to work in both bull and bear markets by taking breakouts in the direction of the daily trend.
"""

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels: R3, R4, S3, S4"""
    range_ = high - low
    close_val = close
    r3 = close_val + range_ * 1.1 / 4
    r4 = close_val + range_ * 1.1 / 2
    s3 = close_val - range_ * 1.1 / 4
    s4 = close_val - range_ * 1.1 / 2
    return r3, r4, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for trend and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    daily_volume = df_1d['volume'].values
    
    # Calculate daily EMA34 for trend filter
    close_series = pd.Series(daily_close)
    ema34_daily = close_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate Camarilla levels from previous day
    r3, r4, s3, s4 = calculate_camarilla(daily_high, daily_low, daily_close)
    
    # Align all to 12h timeframe
    ema34_12h = align_htf_to_ltf(prices, df_1d, ema34_daily)
    r3_12h = align_htf_to_ltf(prices, df_1d, r3)
    r4_12h = align_htf_to_ltf(prices, df_1d, r4)
    s3_12h = align_htf_to_ltf(prices, df_1d, s3)
    s4_12h = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume confirmation: > 1.5x 20-period average on 12h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):  # Start after EMA warmup
        if position == 0:
            # LONG: Price breaks above R3 with volume confirmation and daily uptrend
            if close[i] > r3_12h[i] and close[i-1] <= r3_12h[i-1] and volume_confirm[i] and close[i] > ema34_12h[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 with volume confirmation and daily downtrend
            elif close[i] < s3_12h[i] and close[i-1] >= s3_12h[i-1] and volume_confirm[i] and close[i] < ema34_12h[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches R4 (take profit) or breaks below S3 (stop)
            if close[i] >= r4_12h[i] or close[i] < s3_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches S4 (take profit) or breaks above R3 (stop)
            if close[i] <= s4_12h[i] or close[i] > r3_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals