#!/usr/bin/env python3
"""
6h_WeeklyPivot_DailyTrend_Volume
Hypothesis: Combine weekly pivot points (from 1w) for structural bias with daily EMA trend filter and volume confirmation on 6h.
- Long when price breaks above weekly R1 with daily uptrend and volume spike
- Short when price breaks below weekly S1 with daily downtrend and volume spike
- Exit on opposite pivot level (S1 for long, R1 for short) or trend reversal
Weekly pivots provide strong support/resistance; daily trend filters avoid counter-trend trades; volume confirms conviction.
Designed for low frequency (12-30 trades/year) to minimize fee impact in 6s timeframe.
"""

name = "6h_WeeklyPivot_DailyTrend_Volume"
timeframe = "6h"
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
    
    # Daily trend filter: EMA50
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_daily = close > ema_50
    downtrend_daily = close < ema_50
    
    # Weekly data for pivot points (calculate once per week)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points: P = (H+L+C)/3
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Avoid division by zero or invalid calculations
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s1 = 2 * weekly_pivot - weekly_high
    
    # Align weekly pivots to 6h timeframe (wait for weekly close)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_conf = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get current values
        price = close[i]
        r1 = r1_aligned[i]
        s1 = s1_aligned[i]
        uptrend = uptrend_daily[i]
        downtrend = downtrend_daily[i]
        vol_ok = volume_conf[i]
        
        if position == 0:
            # LONG: break above weekly R1, daily uptrend, volume confirmation
            if price > r1 and uptrend and vol_ok:
                signals[i] = 0.25
                position = 1
            # SHORT: break below weekly S1, daily downtrend, volume confirmation
            elif price < s1 and downtrend and vol_ok:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price touches weekly S1 or daily trend turns down
            if price < s1 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price touches weekly R1 or daily trend turns up
            if price > r1 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals