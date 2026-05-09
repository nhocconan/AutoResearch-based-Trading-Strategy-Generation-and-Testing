#!/usr/bin/env python3
# Hypothesis: 1d timeframe with weekly pivot structure (Camarilla R4/S4) and weekly trend filter.
# Uses weekly Camarilla levels (R4/S4) for breakout entries and weekly EMA21 for trend filter.
# Weekly pivot provides robust structural support/resistance that works in both bull and bear markets.
# Weekly trend filter reduces whipsaw by only allowing trades in direction of higher timeframe trend.
# Target: 30-100 total trades over 4 years (7-25/year) with size 0.25.

name = "1d_Camarilla_R4_S4_1wEMA21_Trend"
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
    
    # Calculate weekly Camarilla levels (R4, S4) from previous week
    # 1w = 480 minutes = 96 * 5min bars, but we use daily data: 1w = 7 days
    prev_close = np.roll(close, 7)  # 7 days back
    prev_high = np.roll(high, 7)
    prev_low = np.roll(low, 7)
    prev_close[:7] = np.nan  # First values invalid
    
    camarilla_range = prev_high - prev_low
    r4 = prev_close + 1.1 * camarilla_range / 2  # R4 level
    s4 = prev_close - 1.1 * camarilla_range / 2  # S4 level
    
    # Breakout conditions: price must close beyond the level
    breakout_up = close > r4
    breakout_down = close < s4
    
    # Get weekly data for EMA21 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    # Calculate 1w EMA21 trend filter
    ema_21_1w = pd.Series(df_1w['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    trend_up = close > ema_21_1w_aligned
    trend_down = close < ema_21_1w_aligned
    
    # Volume filter: current volume > 1.5x 20-period average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Need enough data for indicators
    
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
            # Long: breakout above R4 + 1w uptrend + volume filter
            if breakout_up[i] and trend_up[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: breakout below S4 + 1w downtrend + volume filter
            elif breakout_down[i] and trend_down[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to previous week's close or trend reversal
            if close[i] <= prev_close[i] or not trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to previous week's close or trend reversal
            if close[i] >= prev_close[i] or not trend_down[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals