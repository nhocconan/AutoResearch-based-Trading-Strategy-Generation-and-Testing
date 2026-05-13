#!/usr/bin/env python3
"""
1d_WeeklyPivot_Breakout_Trend_Volume
Hypothesis: Weekly pivot breakouts with 1d trend and volume confirmation work in both bull and bear markets.
Breakout above weekly R1 with uptrend and volume spike = long.
Breakdown below weekly S1 with downtrend and volume spike = short.
Exit on opposite pivot level touch or trend reversal. Weekly pivot levels act as dynamic support/resistance.
Target: 10-25 trades/year per symbol.
"""

name = "1d_WeeklyPivot_Breakout_Trend_Volume"
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
    
    # 1d trend: EMA50
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_1d = close > ema_50
    downtrend_1d = close < ema_50
    
    # Weekly pivot points from 1w data
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    # Calculate pivot points: P = (H+L+C)/3, R1 = 2*P - L, S1 = 2*P - H
    high_w = df_1w['high'].values
    low_w = df_1w['low'].values
    close_w = df_1w['close'].values
    pivot = (high_w + low_w + close_w) / 3.0
    r1 = 2 * pivot - low_w
    s1 = 2 * pivot - high_w
    # Align weekly pivot levels to daily timeframe (wait for weekly bar to close)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_conf = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get values
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        uptrend = uptrend_1d[i]
        downtrend = downtrend_1d[i]
        vol_conf = volume_conf[i]
        
        if position == 0:
            # LONG: break above weekly R1, uptrend, volume confirmation
            if close[i] > r1_val and uptrend and vol_conf:
                signals[i] = 0.25
                position = 1
            # SHORT: break below weekly S1, downtrend, volume confirmation
            elif close[i] < s1_val and downtrend and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: touch weekly S1 or trend turns down
            if close[i] < s1_val or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: touch weekly R1 or trend turns up
            if close[i] > r1_val or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals