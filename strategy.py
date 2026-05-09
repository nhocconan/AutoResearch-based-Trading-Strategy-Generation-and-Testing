#!/usr/bin/env python3
# Hypothesis: 6h timeframe with weekly pivot structure and daily trend filter.
# Uses weekly Camarilla levels (R3/S3) for breakout entries and daily EMA34 for trend filter.
# Weekly pivot provides robust structural support/resistance that works in both bull and bear markets.
# Daily trend filter reduces whipsaw by only allowing trades in direction of higher timeframe trend.
# Target: 50-150 total trades over 4 years (12-37/year) with size 0.25.

name = "6h_Camarilla_R3_S3_1dEMA34_Trend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prrices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate weekly Camarilla levels (R3, S3) from previous week
    # 28 bars = 7 days * 4 bars per day (assuming 6h timeframe: 4 bars per day)
    prev_close = np.roll(close, 28)  # 28 bars = 7 days
    prev_high = np.roll(high, 28)
    prev_low = np.roll(low, 28)
    prev_close[:28] = np.nan  # First values invalid
    
    camarilla_range = prev_high - prev_low
    r3 = prev_close + 1.1 * camarilla_range / 2
    s3 = prev_close - 1.1 * camarilla_range / 2
    
    # Breakout conditions: price must close beyond the level (not just touch)
    breakout_up = close > r3
    breakout_down = close < s3
    
    # Get daily data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    trend_up = close > ema_34_1d_aligned
    trend_down = close < ema_34_1d_aligned
    
    # Volume filter: current volume > 2.0x 20-period average volume (balanced to avoid overtrading)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * avg_volume)
    
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
            # Long: breakout above R3 + 1d uptrend + volume spike
            if breakout_up[i] and trend_up[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: breakout below S3 + 1d downtrend + volume spike
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