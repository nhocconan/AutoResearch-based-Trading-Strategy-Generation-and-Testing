#!/usr/bin/env python3
# Hypothesis: 1d timeframe with weekly Donchian channel breakout and weekly trend filter.
# Uses weekly Donchian(20) for breakout entries and weekly EMA34 for trend filter.
# Weekly Donchian provides structural breakout levels that work in both bull and bear markets.
# Weekly trend filter reduces whipsaw by only allowing trades in direction of higher timeframe trend.
# Target: 20-50 total trades over 4 years (5-12/year) with size 0.25.

name = "1d_Donchian20_1wEMA34_Trend_Volume"
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
    
    # Calculate weekly Donchian channel (20-period) from previous week
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly high and low for Donchian channel
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    
    # Calculate 20-period Donchian bands on weekly data
    upper_band = pd.Series(weekly_high).rolling(window=20, min_periods=20).max().values
    lower_band = pd.Series(weekly_low).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian bands to daily timeframe
    upper_band_aligned = align_htf_to_ltf(prices, df_1w, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_1w, lower_band)
    
    # Breakout conditions: price must close beyond the weekly Donchian bands
    breakout_up = close > upper_band_aligned
    breakout_down = close < lower_band_aligned
    
    # Weekly EMA34 trend filter
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    trend_up = close > ema_34_1w_aligned
    trend_down = close < ema_34_1w_aligned
    
    # Volume filter: current volume > 1.5x 20-period average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
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
            # Long: breakout above weekly upper band + weekly uptrend + volume filter
            if breakout_up[i] and trend_up[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: breakout below weekly lower band + weekly downtrend + volume filter
            elif breakout_down[i] and trend_down[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to weekly lower band or trend reversal
            if close[i] <= lower_band_aligned[i] or not trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to weekly upper band or trend reversal
            if close[i] >= upper_band_aligned[i] or not trend_down[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals