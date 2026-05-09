#!/usr/bin/env python3
# Hypothesis: 1d timeframe with weekly support/resistance levels and daily trend filter.
# Uses weekly pivot points (S1/R1) from prior week for entry/exit and daily EMA34 for trend filter.
# Weekly pivot provides robust structural support/resistance that works in both bull and bear markets.
# Daily trend filter reduces whipsaw by only allowing trades in direction of higher timeframe trend.
# Target: 30-100 total trades over 4 years (7-25/year) with size 0.25.

name = "1d_Pivot_S1_R1_EMA34_Trend_Filter"
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
    
    # Calculate weekly pivot points (S1, R1) from previous week
    # Weekly bars: 7 days * 24h/day = 168 hours, but we need 7 daily bars
    # Since we're on 1d timeframe, 1 week = 7 bars
    prev_week_close = np.roll(close, 7)
    prev_week_high = np.roll(high, 7)
    prev_week_low = np.roll(low, 7)
    prev_week_close[:7] = np.nan  # First values invalid
    
    # Calculate pivot point and support/resistance levels
    pivot = (prev_week_high + prev_week_low + prev_week_close) / 3
    r1 = 2 * pivot - prev_week_low
    s1 = 2 * pivot - prev_week_high
    
    # Breakout conditions: price must close beyond the level
    breakout_up = close > r1
    breakout_down = close < s1
    
    # Get daily data for EMA34 trend filter (using 1d data from itself)
    # But we need to ensure we're using completed daily data
    # Since we're already on 1d timeframe, we can use the current data directly
    # However, we need to make sure we don't use today's close for today's EMA
    # So we'll use a 1-day lag for the EMA calculation to avoid look-ahead
    
    # Calculate EMA34 on close prices with 1-day lag to avoid look-ahead
    close_lagged = np.roll(close, 1)
    close_lagged[0] = np.nan
    
    # Calculate EMA34 with proper handling
    close_series = pd.Series(close_lagged)
    ema_34 = close_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    trend_up = close > ema_34
    trend_down = close < ema_34
    
    # Volume filter: current volume > 1.5x 20-day average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Need enough data for indicators (max of 20 for volume, 34 for EMA, 7 for weekly)
    
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
            # Long: breakout above R1 + daily uptrend + volume filter
            if breakout_up[i] and trend_up[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: breakout below S1 + daily downtrend + volume filter
            elif breakout_down[i] and trend_down[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to weekly pivot or trend reversal
            if close[i] <= pivot[i] or not trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to weekly pivot or trend reversal
            if close[i] >= pivot[i] or not trend_down[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals