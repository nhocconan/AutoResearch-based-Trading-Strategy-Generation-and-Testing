#!/usr/bin/env python3
"""
1d_Camarilla_Pivot_Breakout_WeeklyTrend_Volume
Hypothesis: Camarilla pivot levels (R1/S1) breakouts on daily chart with weekly trend filter and volume confirmation work in both bull and bear markets.
Breakout above R1 with weekly uptrend and volume spike = long.
Breakdown below S1 with weekly downtrend and volume spike = short.
Exit when price touches opposite pivot level (S1 for longs, R1 for shorts) or weekly trend reverses.
Uses weekly trend as higher timeframe filter to avoid counter-trend trades.
Target: 15-25 trades/year per symbol (60-100 total over 4 years).
"""

name = "1d_Camarilla_Pivot_Breakout_WeeklyTrend_Volume"
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
    
    # Calculate previous day's Camarilla pivot levels
    # R1 = close + 1.1 * (high - low) / 12
    # S1 = close - 1.1 * (high - low) / 12
    # We use previous day's values to avoid look-ahead
    prev_high = np.concatenate([[high[0]], high[:-1]])
    prev_low = np.concatenate([[low[0]], low[:-1]])
    prev_close = np.concatenate([[close[0]], close[:-1]])
    
    pivot_range = prev_high - prev_low
    r1 = prev_close + 1.1 * pivot_range / 12
    s1 = prev_close - 1.1 * pivot_range / 12
    
    # Weekly trend filter: EMA50 on weekly data
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_1w = df_1w['close'].values > ema_50_1w
    downtrend_1w = df_1w['close'].values < ema_50_1w
    uptrend_1w_aligned = align_htf_to_ltf(prices, df_1w, uptrend_1w)
    downtrend_1w_aligned = align_htf_to_ltf(prices, df_1w, downtrend_1w)
    
    # Volume confirmation: volume > 1.5 * 20-day average
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_conf = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Get values
        r1_level = r1[i]
        s1_level = s1[i]
        uptrend = uptrend_1w_aligned[i]
        downtrend = downtrend_1w_aligned[i]
        vol_conf = volume_conf[i]
        
        if position == 0:
            # LONG: break above R1, weekly uptrend, volume confirmation
            if close[i] > r1_level and uptrend and vol_conf:
                signals[i] = 0.25
                position = 1
            # SHORT: break below S1, weekly downtrend, volume confirmation
            elif close[i] < s1_level and downtrend and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: touch S1 or weekly trend turns down
            if close[i] < s1_level or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: touch R1 or weekly trend turns up
            if close[i] > r1_level or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals