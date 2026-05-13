#!/usr/bin/env python3
# 1h_Camarilla_R3_S3_Breakout_Trend_Volume_Session
# Hypothesis: Daily Camarilla R3/S3 levels act as strong support/resistance on 1h chart.
# Breakouts above R3 or below S3 with volume confirmation and 4h EMA trend filter capture momentum.
# Uses 4h trend for direction and 1h for precise entry timing. Session filter (08-20 UTC) reduces noise.
# Target ~20-50 trades/year to avoid fee drag. Works in bull (breakouts with trend) and bear (breakdowns against trend filtered by 4h EMA).

name = "1h_Camarilla_R3_S3_Breakout_Trend_Volume_Session"
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
    volume = prices['volume'].values
    
    # Get daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily Camarilla levels: R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    camarilla_r3 = c_1d + (h_1d - l_1d) * 1.1 / 2.0
    camarilla_s3 = c_1d - (h_1d - l_1d) * 1.1 / 2.0
    
    # Align daily Camarilla levels to 1h chart (wait for daily close)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Get 4h data for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    # Session filter: 08:00-20:00 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup
        if not session_filter[i]:
            signals[i] = 0.0
            continue
            
        if position == 0:
            # LONG: Breakout above daily R3 with volume confirmation and 4h EMA uptrend
            if close[i] > camarilla_r3_aligned[i] and volume_filter[i] and close[i] > ema_4h_aligned[i]:
                signals[i] = 0.20
                position = 1
            # SHORT: Breakdown below daily S3 with volume confirmation and 4h EMA downtrend
            elif close[i] < camarilla_s3_aligned[i] and volume_filter[i] and close[i] < ema_4h_aligned[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to daily S3 or breaks below 4h EMA
            if close[i] < camarilla_s3_aligned[i] or close[i] < ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price returns to daily R3 or breaks above 4h EMA
            if close[i] > camarilla_r3_aligned[i] or close[i] > ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals