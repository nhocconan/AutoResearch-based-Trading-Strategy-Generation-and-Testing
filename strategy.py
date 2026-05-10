#!/usr/bin/env python3
# 1d_Weekly_Trend_Follow_With_Daily_Pullback
# Hypothesis: In trending markets (determined by weekly trend), price pulls back to daily support/resistance
# provide high-probability entry points. We use weekly EMA20 for trend direction and enter when
# price touches the daily 20-period EMA in the direction of the weekly trend. This strategy
# aims to capture trend continuation with low-frequency entries to minimize fee drag.
# Target: 10-25 trades/year to stay well under limits.

name = "1d_Weekly_Trend_Follow_With_Daily_Pullback"
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA20 for trend
    ema_20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate daily EMA20 for entry signals
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need weekly EMA20 (20) and daily EMA20 (20)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_20_1w_aligned[i]) or 
            np.isnan(ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly trend filter
        weekly_uptrend = close[i] > ema_20_1w_aligned[i]
        weekly_downtrend = close[i] < ema_20_1w_aligned[i]
        
        if position == 0:
            # Long entry: weekly uptrend + price touches/pulls back to daily EMA20
            if weekly_uptrend and low[i] <= ema_20[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: weekly downtrend + price touches/pulls back to daily EMA20
            elif weekly_downtrend and high[i] >= ema_20[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: weekly trend turns down or price moves significantly above EMA
            if not weekly_uptrend or close[i] > ema_20[i] * 1.05:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: weekly trend turns up or price moves significantly below EMA
            if not weekly_downtrend or close[i] < ema_20[i] * 0.95:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals