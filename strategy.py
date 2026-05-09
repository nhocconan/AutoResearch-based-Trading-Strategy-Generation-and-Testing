#!/usr/bin/env python3
# Hypothesis: 6h timeframe with weekly pivot structure and weekly trend filter.
# Uses weekly Camarilla levels (R3/S3) for breakout entries and weekly EMA34 for trend filter.
# Weekly pivot provides robust structural support/resistance that works in both bull and bear markets.
# Weekly trend filter reduces whipsaw by only allowing trades in direction of higher timeframe trend.
# Target: 50-150 total trades over 4 years (12-37/year) with size 0.25.

name = "6h_Camarilla_R3_S3_1wEMA34_Trend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate weekly data for indicators
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate weekly EMA34 trend filter
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate weekly Camarilla levels (R3, S3) from previous week
    prev_close_1w = np.roll(df_1w['close'].values, 1)
    prev_high_1w = np.roll(df_1w['high'].values, 1)
    prev_low_1w = np.roll(df_1w['low'].values, 1)
    prev_close_1w[0] = np.nan
    
    camarilla_range_1w = prev_high_1w - prev_low_1w
    r3 = prev_close_1w + 1.1 * camarilla_range_1w * 1.125  # R3 = C + 1.1*range*1.125
    s3 = prev_close_1w - 1.1 * camarilla_range_1w * 1.125  # S3 = C - 1.1*range*1.125
    
    # Align weekly levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    # Breakout conditions: price must close beyond the level (not just touch)
    breakout_up = close > r3_aligned
    breakout_down = close < s3_aligned
    
    # Trend filter: price relative to weekly EMA34
    trend_up = close > ema_34_1w_aligned
    trend_down = close < ema_34_1w_aligned
    
    # Volume filter: current volume > 1.5x 20-period average volume (conservative to avoid overtrading)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Need enough data for indicators
    
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
            # Long: breakout above R3 + weekly uptrend + volume filter
            if breakout_up[i] and trend_up[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: breakout below S3 + weekly downtrend + volume filter
            elif breakout_down[i] and trend_down[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to weekly EMA34 or trend reversal
            if close[i] <= ema_34_1w_aligned[i] or not trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to weekly EMA34 or trend reversal
            if close[i] >= ema_34_1w_aligned[i] or not trend_down[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals