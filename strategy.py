#!/usr/bin/env python3

# Hypothesis: 12h timeframe with 1-day Donchian channel breakout and 1-week EMA trend filter.
# Uses daily Donchian(20) for breakout entries and 1-week EMA50 for trend filter.
# Daily Donchian provides robust breakout levels that work in trending markets,
# while weekly EMA filters for the dominant trend to avoid counter-trend trades.
# Volume confirmation ensures breakouts have conviction.
# Target: 50-150 total trades over 4 years (12-37/year) with size 0.25.
# Weekly trend filter reduces whipsaw and improves win rate in ranging markets.

name = "12h_Donchian20_1wEMA50_Trend_Volume"
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
    
    # Get daily data for Donchian channel
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily Donchian(20) - using previous day's data to avoid look-ahead
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    
    # Upper band: highest high of last 20 days (excluding current day)
    upper_band = pd.Series(daily_high).rolling(window=20, min_periods=20).max().shift(1).values
    # Lower band: lowest low of last 20 days (excluding current day)
    lower_band = pd.Series(daily_low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Align Donchian bands to 12h timeframe
    upper_band_aligned = align_htf_to_ltf(prices, df_1d, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_1d, lower_band)
    
    # Breakout conditions: price must close beyond the level
    breakout_up = close > upper_band_aligned
    breakout_down = close < lower_band_aligned
    
    # Get weekly data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    trend_up = close > ema_50_1w_aligned
    trend_down = close < ema_50_1w_aligned
    
    # Volume filter: current volume > 1.5x 20-period average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(breakout_up[i]) or np.isnan(breakout_down[i]) or
            np.isnan(trend_up[i]) or np.isnan(trend_down[i]) or
            np.isnan(volume_filter[i]) or
            np.isnan(upper_band_aligned[i]) or np.isnan(lower_band_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: breakout above upper band + 1w uptrend + volume spike
            if breakout_up[i] and trend_up[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: breakout below lower band + 1w downtrend + volume spike
            elif breakout_down[i] and trend_down[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to lower band or trend reversal
            if close[i] <= lower_band_aligned[i] or not trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to upper band or trend reversal
            if close[i] >= upper_band_aligned[i] or not trend_down[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals