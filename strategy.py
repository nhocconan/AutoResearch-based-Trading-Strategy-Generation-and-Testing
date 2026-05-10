#!/usr/bin/env python3
# 1d_WeeklyTrend_Follow_With_DailyPullback
# Hypothesis: In multi-year crypto markets, strong weekly trends provide reliable directional bias.
# We use weekly EMA20 as the primary trend filter and enter on daily pullbacks to daily EMA20.
# Long when: weekly trend up (weekly close > weekly EMA20) AND daily price pulls back to touch daily EMA20 from below.
# Short when: weekly trend down (weekly close < weekly EMA20) AND daily price pulls back to touch daily EMA20 from above.
# This captures continuation moves in trending markets while avoiding counter-trend trades.
# Works in both bull (follows strong uptrends) and bear (follows strong downtrends).
# Uses volume confirmation to avoid low-conviction breakouts.

name = "1d_WeeklyTrend_Follow_With_DailyPullback"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate daily EMA20 for pullback entries
    close_s = pd.Series(close)
    ema_20 = close_s.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate weekly EMA20 for trend filter
    ema_20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume confirmation (20-day MA on daily chart)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA20 (20), weekly EMA20 (20), volume MA (20)
    start_idx = max(20, 20, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_20[i]) or np.isnan(ema_20_1w_aligned[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly trend filter
        uptrend = close[i] > ema_20_1w_aligned[i]
        downtrend = close[i] < ema_20_1w_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        # Price relative to daily EMA20 for pullback detection
        if i > 0:
            cross_above_ema20 = (close[i] > ema_20[i]) and (close[i-1] <= ema_20[i-1])
            cross_below_ema20 = (close[i] < ema_20[i]) and (close[i-1] >= ema_20[i-1])
        else:
            cross_above_ema20 = False
            cross_below_ema20 = False
        
        if position == 0:
            # Long entry: weekly uptrend + pullback to daily EMA20 from below + volume
            if uptrend and cross_above_ema20 and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: weekly downtrend + pullback to daily EMA20 from above + volume
            elif downtrend and cross_below_ema20 and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: weekly trend breaks or reversal signal
            if not uptrend or cross_below_ema20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: weekly trend breaks or reversal signal
            if not downtrend or cross_above_ema20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals