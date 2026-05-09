# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# Hypothesis: 4h timeframe with daily pivot structure and weekly trend filter.
# Uses daily Camarilla levels (R4/S4) for breakout entries and weekly EMA50 for trend filter.
# Daily pivot provides robust support/resistance levels that work in both bull and bear markets.
# Weekly trend filter reduces whipsaw by only allowing trades in direction of higher timeframe trend.
# Volume confirmation ensures breakouts have institutional participation.
# Target: 75-200 total trades over 4 years (19-50/year) with size 0.25.

name = "4h_Camarilla_R4_S4_1wEMA50_Trend_Volume"
timeframe = "4h"
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
    
    # Calculate daily Camarilla levels (R4, S4) from previous day
    prev_close = np.roll(close, 6)   # 6 bars = 1.5 days * 4 bars per day (approximates daily)
    prev_high = np.roll(high, 6)
    prev_low = np.roll(low, 6)
    prev_close[:6] = np.nan  # First values invalid
    
    camarilla_range = prev_high - prev_low
    r4 = prev_close + 1.1 * camarilla_range / 2
    s4 = prev_close - 1.1 * camarilla_range / 2
    
    # Breakout conditions: price must close beyond the level
    breakout_up = close > r4
    breakout_down = close < s4
    
    # Get weekly data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    trend_up = close > ema_50_1w_aligned
    trend_down = close < ema_50_1w_aligned
    
    # Volume filter: current volume > 1.8x 24-period average volume (balanced to avoid overtrading)
    avg_volume = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (1.8 * avg_volume)
    
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
            # Long: breakout above R4 + weekly uptrend + volume spike
            if breakout_up[i] and trend_up[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: breakout below S4 + weekly downtrend + volume spike
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