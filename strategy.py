#!/usr/bin/env python3

# Hypothesis: 12h timeframe with daily pivot structure (from 1d) and hourly trend filter.
# Uses daily Camarilla levels (R1/S1) for breakout entries and 1h EMA20 for trend filter.
# Daily pivot provides clean support/resistance levels that work in both bull and bear markets.
# Target: 50-150 total trades over 4 years (12-37/year) with size 0.25.
# 12h timeframe reduces noise and improves win rate while maintaining sufficient trade frequency.

name = "12h_Camarilla_R1_S1_1hEMA20_Trend_Volume"
timeframe = "12h"
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
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = np.nan  # First value invalid
    
    camarilla_range = prev_high - prev_low
    r1 = prev_close + 1.1 * camarilla_range / 4
    s1 = prev_close - 1.1 * camarilla_range / 4
    
    # Breakout conditions: price must close beyond the level (not just touch)
    breakout_up = close > r1
    breakout_down = close < s1
    
    # Get hourly data for EMA20 trend filter
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 20:
        return np.zeros(n)
    
    # Calculate 1h EMA20 trend filter
    ema_20_1h = pd.Series(df_1h['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1h_aligned = align_htf_to_ltf(prices, df_1h, ema_20_1h)
    
    trend_up = close > ema_20_1h_aligned
    trend_down = close < ema_20_1h_aligned
    
    # Volume filter: current volume > 1.5x 24-period average volume (12h = 24*30m bars)
    avg_volume = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Need enough data for indicators
    
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
            # Long: breakout above R1 + 1h uptrend + volume spike
            if breakout_up[i] and trend_up[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: breakout below S1 + 1h downtrend + volume spike
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