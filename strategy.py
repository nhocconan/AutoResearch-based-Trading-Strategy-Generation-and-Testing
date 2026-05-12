#!/usr/bin/env python3
# 1D_DONCHIAN20_BREAKOUT_1W_TREND_FILTER
# Hypothesis: Donchian channel breakouts on daily timeframe capture strong trends.
# Weekly trend filter ensures we only trade in the direction of the higher timeframe trend,
# reducing false signals and whipsaws. Works in both bull and bear markets:
# - In bull markets: captures breakout continuations in uptrend
# - In bear markets: captures breakdown continuations in downtrend
# Target: 10-25 trades/year on 1d timeframe.

name = "1D_DONCHIAN20_BREAKOUT_1W_TREND_FILTER"
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
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly EMA34 for trend filter
    ema34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Daily Donchian channel (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure Donchian is calculated
    
    for i in range(start_idx, n):
        # Skip if weekly trend data is not ready
        if np.isnan(ema34_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above upper Donchian band in weekly uptrend
            if high[i] > high_roll[i-1] and close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below lower Donchian band in weekly downtrend
            elif low[i] < low_roll[i-1] and close[i] < ema34_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price falls below lower Donchian band or weekly trend turns down
            if low[i] < low_roll[i-1] or close[i] < ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price rises above upper Donchian band or weekly trend turns up
            if high[i] > high_roll[i-1] or close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals