#!/usr/bin/env python3
"""
1d_WR_Extreme_1wTrend
Williams %R (14) > 0 oversold and < -80 oversold with 1-week EMA trend filter.
Extreme readings in trending markets yield high-probability reversals.
Designed for 1d timeframe with 1-week trend filter to reduce whipsaw and capture swings.
Target: 15-25 trades/year to minimize fee drag.
"""

name = "1d_WR_Extreme_1wTrend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Weekly EMA trend (34-period)
    ema_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Williams %R (14-period) on daily
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    # Avoid division by zero
    rr = highest_high - lowest_low
    wr = np.where(rr != 0, -100 * (highest_high - close) / rr, -50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback, n):
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(wr[i]) or
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: WR oversold (< -80) in weekly uptrend
            if wr[i] < -80 and close[i] > ema_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: WR overbought (> -20) in weekly downtrend
            elif wr[i] > -20 and close[i] < ema_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: WR returns above -50 or trend changes
            if wr[i] > -50 or close[i] < ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: WR returns below -50 or trend changes
            if wr[i] < -50 or close[i] > ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals