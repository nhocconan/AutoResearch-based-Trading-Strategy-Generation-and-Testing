#!/usr/bin/env python3
"""
1d_Weekly_Pivot_R3_S3_Breakout_Trend_Volume
Hypothesis: Weekly pivot levels (R3/S3) act as strong support/resistance. 
Breakouts above R3 or below S3 with weekly trend alignment and volume confirmation 
capture sustained moves. Designed for low frequency (<20/year) to avoid fee drag.
Works in bull (breakouts up) and bear (breakouts down) markets.
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
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly high/low/close for pivot calculation (using 1w data)
    df_1w = get_htf_data(prices, '1w')
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Calculate weekly pivot points: P = (H+L+C)/3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    # R3 = H + 2*(P - L) = 3*P - 2*L
    # S3 = L - 2*(H - P) = 3*L - 2*H
    weekly_r3 = 3 * weekly_pivot - 2 * weekly_low
    weekly_s3 = 3 * weekly_low - 2 * weekly_high
    
    # Align weekly levels to daily timeframe (wait for weekly close)
    r3_aligned = align_htf_to_ltf(prices, df_1w, weekly_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, weekly_s3)
    
    # Weekly trend: EMA50 on weekly close
    weekly_ema50 = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_uptrend = weekly_close > weekly_ema50
    weekly_downtrend = weekly_close < weekly_ema50
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend)
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_downtrend)
    
    # Volume confirmation: > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if position == 0:
            # LONG: Close above weekly R3, weekly uptrend, volume confirmation
            if close[i] > r3_aligned[i] and weekly_uptrend_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Close below weekly S3, weekly downtrend, volume confirmation
            elif close[i] < s3_aligned[i] and weekly_downtrend_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below weekly pivot or trend fails
            if close[i] < weekly_pivot[i] or not weekly_uptrend_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above weekly pivot or trend fails
            if close[i] > weekly_pivot[i] or not weekly_downtrend_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals