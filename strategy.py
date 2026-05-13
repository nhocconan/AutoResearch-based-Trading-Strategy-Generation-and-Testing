#!/usr/bin/env python3
"""
6h_LongTermTrend_Support_Resistance_Bounce
Hypothesis: In trending markets, price tends to respect longer-term support/resistance levels derived from weekly chart (weekly high/low of last 4 weeks). 
Price pulling back to these levels in the direction of the 1-week trend offers high-probability entries. 
Volume confirmation filters out false breaks. Works in both bull (buy dips in uptrend) and bear (sell rallies in downtrend).
Target: 15-35 trades/year per symbol.
"""

name = "6h_LongTermTrend_Support_Resistance_Bounce"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly high/low of last 4 weeks (28 trading days assuming 5 days/week, but using actual weeks)
    # Get weekly data
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 4:
        return np.zeros(n)
    
    # Calculate weekly high and low for the last 4 completed weeks
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    
    # For each point, we want the highest high and lowest low from the last 4 weekly bars (excluding current forming week)
    # We'll compute rolling max/min over 4 weeks on the weekly data, then align
    # But to avoid look-ahead, we use the value from 4 weeks ago to current? Actually, we want the range of the last 4 completed weeks.
    # For simplicity and to avoid look-ahead, we'll use the highest high and lowest low from the prior 4 weeks (weeks t-4 to t-1)
    # So for weekly index i, we look at [i-4, i-1]
    # We'll compute this using a rolling window on the weekly data, but shift by 1 to avoid using current week
    
    # Convert to series for rolling
    weekly_high_series = pd.Series(weekly_high)
    weekly_low_series = pd.Series(weekly_low)
    
    # Rolling max of high over 4 weeks, min of low over 4 weeks, but we want the window of the 4 weeks BEFORE the current week
    # So we compute rolling window on the weekly data, then shift by 1 week to exclude current
    max_high_4w = weekly_high_series.rolling(window=4, min_periods=4).max().shift(1).values
    min_low_4w = weekly_low_series.rolling(window=4, min_periods=4).min().shift(1).values
    
    # Align to 6t timeframe
    max_high_4w_aligned = align_htf_to_ltf(prices, df_1w, max_high_4w)
    min_low_4w_aligned = align_htf_to_ltf(prices, df_1w, min_low_4w)
    
    # 1-week trend: using weekly EMA 8 for trend direction
    ema_8_1w = pd.Series(df_1w['close']).ewm(span=8, adjust=False, min_periods=8).mean().values
    uptrend_1w = df_1w['close'].values > ema_8_1w
    downtrend_1w = df_1w['close'].values < ema_8_1w
    uptrend_1w_aligned = align_htf_to_ltf(prices, df_1w, uptrend_1w)
    downtrend_1w_aligned = align_htf_to_ltf(prices, df_1w, downtrend_1w)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_conf = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index 50 to ensure enough data for weekly alignment (though weekly alignment is handled by align_htf_to_ltf)
    for i in range(50, n):
        # Get values
        res_level = max_high_4w_aligned[i]  # resistance level from last 4 weeks high
        sup_level = min_low_4w_aligned[i]   # support level from last 4 weeks low
        uptrend = uptrend_1w_aligned[i]
        downtrend = downtrend_1w_aligned[i]
        vol_conf = volume_conf[i]
        
        if position == 0:
            # LONG: price near support level (within 0.5%) and weekly uptrend and volume confirmation
            if sup_level > 0 and abs(low[i] - sup_level) / sup_level < 0.005 and uptrend and vol_conf:
                signals[i] = 0.25
                position = 1
            # SHORT: price near resistance level (within 0.5%) and weekly downtrend and volume confirmation
            elif res_level > 0 and abs(high[i] - res_level) / res_level < 0.005 and downtrend and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price reaches midpoint between support and resistance or weekly trend turns down
            mid_level = (sup_level + res_level) / 2.0
            if close[i] >= mid_level or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price reaches midpoint or weekly trend turns up
            mid_level = (sup_level + res_level) / 2.0
            if close[i] <= mid_level or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals