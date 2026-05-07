#!/usr/bin/env python3
# 6h_WeeklyPivot_DonchianBreakout_Volume
# Hypothesis: Weekly pivots define strong support/resistance zones. Price breaking Donchian(20) with volume
# confirmation in the direction of weekly pivot bias captures breakouts with institutional interest.
# Weekly pivot bias: price above weekly pivot = bullish bias (long breakouts), below = bearish bias (short breakouts).
# Works in both bull and bear markets by filtering breakouts with weekly structure and volume.
# Target: 50-150 total trades over 4 years (~12-37/year) to minimize fee drag.
timeframe = "6h"
name = "6h_WeeklyPivot_DonchianBreakout_Volume"
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
    
    # Weekly high, low, close for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Weekly pivot point: (H + L + C) / 3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    # Align weekly pivot to 6h timeframe (wait for weekly close)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Donchian channel (20-period) on 6h
    donch_period = 20
    donch_high = pd.Series(high).rolling(window=donch_period, min_periods=donch_period).max().values
    donch_low = pd.Series(low).rolling(window=donch_period, min_periods=donch_period).min().values
    
    # Volume spike: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(donch_period, n):
        # Skip if any critical value is NaN
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] == 0 or
            np.isnan(weekly_pivot_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high + volume spike + price above weekly pivot (bullish bias)
            if (close[i] > donch_high[i] and 
                volume[i] > 1.5 * vol_ma[i] and
                close[i] > weekly_pivot_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low + volume spike + price below weekly pivot (bearish bias)
            elif (close[i] < donch_low[i] and 
                  volume[i] > 1.5 * vol_ma[i] and
                  close[i] < weekly_pivot_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price closes below Donchian low (stoploss)
            if close[i] < donch_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price closes above Donchian high (stoploss)
            if close[i] > donch_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals