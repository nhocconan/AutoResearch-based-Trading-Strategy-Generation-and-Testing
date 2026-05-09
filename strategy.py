#!/usr/bin/env python3
# Hypothesis: 6h timeframe with weekly pivot structure (from 1w) and daily trend filter.
# Uses weekly Camarilla levels (R4/S4) for breakout entries and daily EMA34 for trend filter.
# Weekly pivot provides stronger structural support/resistance that works in both bull and bear markets.
# Daily trend filter reduces whipsaw by only allowing trades in direction of higher timeframe trend.
# Target: 50-150 total trades over 4 years (12-37/year) with size 0.25.

name = "6h_Camarilla_R4_S4_1dEMA34_Trend_Volume"
timeframe = "6h"
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
    prev_close_w = np.roll(close, 7 * 4)  # 4 periods per day for 6h timeframe (24h/6h=4)
    prev_high_w = np.roll(high, 7 * 4)
    prev_low_w = np.roll(low, 7 * 4)
    prev_close_w[:7*4] = np.nan  # First week invalid
    
    camarilla_range_w = prev_high_w - prev_low_w
    r4 = prev_close_w + 1.1 * camarilla_range_w * 1.5  # R4 = close + 1.1 * range * 1.5
    s4 = prev_close_w - 1.1 * camarilla_range_w * 1.5  # S4 = close - 1.1 * range * 1.5
    
    # Breakout conditions: price must close beyond the level (not just touch)
    breakout_up = close > r4
    breakout_down = close < s4
    
    # Get daily data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    trend_up = close > ema_34_1d_aligned
    trend_down = close < ema_34_1d_aligned
    
    # Volume filter: current volume > 2.0x 20-period average volume
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
            # Long: breakout above R4 + 1d uptrend + volume spike
            if breakout_up[i] and trend_up[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: breakout below S4 + 1d downtrend + volume spike
            elif breakout_down[i] and trend_down[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to previous week's close or trend reversal
            if close[i] <= prev_close_w[i] or not trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to previous week's close or trend reversal
            if close[i] >= prev_close_w[i] or not trend_down[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals