#!/usr/bin/env python3
# 1d_Donchian20_Breakout_1wTrend_Filter
# Hypothesis: On 1d timeframe, enter long when price breaks above 20-day Donchian high
# with weekly trend confirmation (price > weekly SMA50). Enter short when price breaks
# below 20-day Donchian low with price < weekly SMA50. Exit on opposite Donchian break.
# Uses weekly trend filter to avoid counter-trend trades in choppy markets. Targets
# 15-25 trades/year for low fee drift. Works in bull via breakouts and in bear via
# shorting breakdowns with trend filter.

name = "1d_Donchian20_Breakout_1wTrend_Filter"
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
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    weekly_close = df_1w['close'].values
    
    # Weekly SMA50 for trend filter
    sma50_1w = pd.Series(weekly_close).rolling(window=50, min_periods=50).mean().values
    sma50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma50_1w)
    
    # 20-day Donchian channels
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(sma50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above 20-day Donchian high with weekly uptrend
            if close[i] > donch_high[i] and close[i] > sma50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below 20-day Donchian low with weekly downtrend
            elif close[i] < donch_low[i] and close[i] < sma50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below 20-day Donchian low
            if close[i] < donch_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above 20-day Donchian high
            if close[i] > donch_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals