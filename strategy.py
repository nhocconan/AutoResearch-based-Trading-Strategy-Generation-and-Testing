#!/usr/bin/env python3
# Hypothesis: 4h timeframe with daily pivot structure and 12h trend filter.
# Uses daily Camarilla levels (R1/S1) for breakout entries and 12h EMA50 for trend filter.
# Daily pivot provides robust support/resistance that works in both bull and bear markets.
# 12h trend filter reduces whipsaw by only allowing trades in direction of higher timeframe trend.
# Volume confirmation ensures participation. Target: 75-200 total trades over 4 years (19-50/year) with size 0.25.

name = "4h_Camarilla_R1_S1_1dPivot_12hEMA50_Trend_Volume"
timeframe = "4h"
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
    
    # Calculate daily Camarilla levels (R1, S1) from previous day
    prev_close = np.roll(close, 6)   # 6 bars = 1 day * 6 bars per day (4h TF)
    prev_high = np.roll(high, 6)
    prev_low = np.roll(low, 6)
    prev_close[:6] = np.nan          # First values invalid
    
    camarilla_range = prev_high - prev_low
    r1 = prev_close + 1.1 * camarilla_range / 4
    s1 = prev_close - 1.1 * camarilla_range / 4
    
    # Breakout conditions: price must close beyond the level (not just touch)
    breakout_up = close > r1
    breakout_down = close < s1
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    trend_up = close > ema_50_12h_aligned
    trend_down = close < ema_50_12h_aligned
    
    # Volume filter: current volume > 1.5x 20-period average volume (moderate to avoid overtrading)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(breakout_up[i]) or np.isnan(breakout_down[i]) or
            np.isnan(trend_up[i]) or np.isnan(trend_down[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: breakout above R1 + 12h uptrend + volume spike
            if breakout_up[i] and trend_up[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: breakout below S1 + 12h downtrend + volume spike
            elif breakout_down[i] and trend_down[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to previous day's close or trend reversal
            if close[i] <= prev_close[i] or not trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to previous day's close or trend reversal
            if close[i] >= prev_close[i] or not trend_down[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals