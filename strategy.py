#!/usr/bin/env python3
# 1d_WeeklyTrend_Follow_With_DailyPullback
# Hypothesis: Weekly trend filter combined with daily pullback entries provides
# high-probability trend-following trades. Uses 10-week EMA for trend direction and
# enters on pullbacks to the 20-day EMA in the direction of the weekly trend.
# Weekly trend reduces whipsaw, daily EMA provides entry timing. Designed for
# low trade frequency (10-30/year) to minimize fee drag on 1d timeframe.

name = "1d_WeeklyTrend_Follow_With_DailyPullback"
timeframe = "1d"
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA10 for trend filter
    ema_10_1w = pd.Series(df_1w['close']).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema_10_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_10_1w)
    
    # Calculate daily EMA20 for pullback entries
    ema_20_daily = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume confirmation (20-day average)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need weekly EMA10 (20), daily EMA20 (20), volume MA (20)
    start_idx = max(20, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_10_1w_aligned[i]) or 
            np.isnan(ema_20_daily[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly trend filter
        uptrend = close[i] > ema_10_1w_aligned[i]
        downtrend = close[i] < ema_10_1w_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        if position == 0:
            # Long entry: weekly uptrend + price pulls back to or below daily EMA20 + volume
            if uptrend and close[i] <= ema_20_daily[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: weekly downtrend + price pulls back to or above daily EMA20 + volume
            elif downtrend and close[i] >= ema_20_daily[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: weekly trend turns down or price moves above daily EMA20 (take profit)
            if not uptrend or close[i] > ema_20_daily[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: weekly trend turns up or price moves below daily EMA20 (take profit)
            if not downtrend or close[i] < ema_20_daily[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals