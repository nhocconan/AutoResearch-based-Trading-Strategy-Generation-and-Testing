#!/usr/bin/env python3
"""
1d_WeeklyDonchian_Breakout_Trend_Volume_v3
Hypothesis: Weekly Donchian channel breakouts with daily trend (EMA50) and volume confirmation capture major trends in both bull and bear markets. Weekly timeframe reduces noise and false breakouts, while daily trend filter ensures alignment with intermediate trend. Volume confirmation adds conviction. Designed for low trade frequency to minimize fee drag in ranging markets.
"""

name = "1d_WeeklyDonchian_Breakout_Trend_Volume_v3"
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
    
    # Weekly Donchian Channel: 20-week high/low
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate 20-period rolling high/low on weekly data
    roll_high = np.zeros(len(high_1w))
    roll_low = np.zeros(len(low_1w))
    for i in range(20, len(high_1w)):
        roll_high[i] = np.max(high_1w[i-20:i])
        roll_low[i] = np.min(low_1w[i-20:i])
    # For first 20 periods, use expanding window
    for i in range(20):
        roll_high[i] = np.max(high_1w[:i+1])
        roll_low[i] = np.min(low_1w[:i+1])
    
    # Align to daily timeframe (with proper delay for weekly bar close)
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, roll_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, roll_low)
    
    # Daily trend filter: EMA50
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_daily = close > ema_50
    downtrend_daily = close < ema_50
    
    # Volume confirmation: volume > 1.5 * 20-day average
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    for i in range(20):
        vol_ma[i] = np.mean(volume[:i+1])
    volume_conf = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get values
        weekly_high = weekly_high_aligned[i]
        weekly_low = weekly_low_aligned[i]
        uptrend = uptrend_daily[i]
        downtrend = downtrend_daily[i]
        vol_conf = volume_conf[i]
        
        if position == 0:
            # LONG: break above weekly high, daily uptrend, volume confirmation
            if close[i] > weekly_high and uptrend and vol_conf:
                signals[i] = 0.25
                position = 1
            # SHORT: break below weekly low, daily downtrend, volume confirmation
            elif close[i] < weekly_low and downtrend and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: touch weekly low or daily trend turns down
            if close[i] < weekly_low or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: touch weekly high or daily trend turns up
            if close[i] > weekly_high or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals