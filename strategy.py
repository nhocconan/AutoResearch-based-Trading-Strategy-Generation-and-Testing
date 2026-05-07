#!/usr/bin/env python3
# 1d_WeeklyTrend_DailyBreakout_Volume
# Hypothesis: Daily chart strategy using weekly trend filter with daily price breakouts and volume confirmation.
# Uses weekly EMA to determine trend direction, then looks for breakouts of daily high/low with volume confirmation.
# Designed to work in both bull and bear markets by only taking trades in the direction of the weekly trend.
# Target: 10-25 trades/year per symbol to minimize fee drag while maintaining edge.

name = "1d_WeeklyTrend_DailyBreakout_Volume"
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # Calculate weekly EMA21 for trend filter
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Daily high/low for breakout levels (using previous day's values)
    daily_high = np.maximum.accumulate(high)
    daily_low = np.minimum.accumulate(low)
    
    # Volume spike detection: 2x average volume (20-period = ~1 month on daily)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Ensure we have enough data for weekly EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(ema_21_1w_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine weekly trend: price above/below weekly EMA21
        weekly_uptrend = close[i] > ema_21_1w_aligned[i]
        weekly_downtrend = close[i] < ema_21_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above previous day's high with volume, in weekly uptrend
            if (high[i] > daily_high[i-1] and 
                volume[i] > 2.0 * vol_ma[i] and 
                weekly_uptrend):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below previous day's low with volume, in weekly downtrend
            elif (low[i] < daily_low[i-1] and 
                  volume[i] > 2.0 * vol_ma[i] and 
                  weekly_downtrend):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price breaks below previous day's low (reversal signal)
            if low[i] < daily_low[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks above previous day's high (reversal signal)
            if high[i] > daily_high[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals