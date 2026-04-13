#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Hypothesis: 6h Donchian(20) breakout with weekly trend filter
    # Enter long when price breaks above 20-bar high AND weekly close > weekly open (bullish week)
    # Enter short when price breaks below 20-bar low AND weekly close < weekly open (bearish week)
    # Exit on opposite Donchian breakout (10-bar) or weekly trend reversal
    # Weekly trend filter ensures we only trade with the dominant weekly momentum
    # Donchian breakouts capture strong moves; weekly filter avoids counter-trend whipsaws
    # Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 6h data for primary timeframe
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Get 1w data for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    open_1w = df_1w['open'].values
    
    # Calculate 6h Donchian channels (20-bar for entry, 10-bar for exit)
    # Donchian high = max(high over period), Donchian low = min(low over period)
    donchian_high_20 = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    donchian_high_10 = pd.Series(high_6h).rolling(window=10, min_periods=10).max().values
    donchian_low_10 = pd.Series(low_6h).rolling(window=10, min_periods=10).min().values
    
    # Align Donchian levels to 6h timeframe (already aligned as we used 6h data)
    # But we need to shift by 1 to avoid look-ahead (use previous bar's levels)
    donchian_high_20_aligned = np.roll(donchian_high_20, 1)
    donchian_low_20_aligned = np.roll(donchian_low_20, 1)
    donchian_high_10_aligned = np.roll(donchian_high_10, 1)
    donchian_low_10_aligned = np.roll(donchian_low_10, 1)
    # Set first value to NaN (no prior data)
    donchian_high_20_aligned[0] = np.nan
    donchian_low_20_aligned[0] = np.nan
    donchian_high_10_aligned[0] = np.nan
    donchian_low_10_aligned[0] = np.nan
    
    # Calculate weekly trend: bullish if weekly close > weekly open, bearish if <
    weekly_bullish = close_1w > open_1w
    weekly_bearish = close_1w < open_1w
    
    # Align weekly trend to 6h timeframe
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):  # start from 20 to have enough data for Donchian
        # Skip if data not ready
        if (np.isnan(donchian_high_20_aligned[i]) or np.isnan(donchian_low_20_aligned[i]) or
            np.isnan(donchian_high_10_aligned[i]) or np.isnan(donchian_low_10_aligned[i]) or
            np.isnan(weekly_bullish_aligned[i]) or np.isnan(weekly_bearish_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Donchian breakout conditions (using current bar's close vs previous bar's levels)
        breakout_up = close[i] > donchian_high_20_aligned[i]  # break above 20-bar high
        breakout_down = close[i] < donchian_low_20_aligned[i]  # break below 20-bar low
        
        # Exit conditions: opposite 10-bar Donchian breakout or weekly trend reversal
        exit_long = close[i] < donchian_low_10_aligned[i]  # break below 10-bar low
        exit_short = close[i] > donchian_high_10_aligned[i]  # break above 10-bar high
        weekly_reversal_long = weekly_bearish_aligned[i]  # weekly turned bearish
        weekly_reversal_short = weekly_bullish_aligned[i]  # weekly turned bullish
        
        # Entry conditions with weekly trend filter
        long_entry = breakout_up and weekly_bullish_aligned[i] and position != 1
        short_entry = breakout_down and weekly_bearish_aligned[i] and position != -1
        
        # Execute signals
        if long_entry:
            position = 1
            signals[i] = position_size
        elif short_entry:
            position = -1
            signals[i] = -position_size
        elif position == 1 and (exit_long or weekly_reversal_long):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (exit_short or weekly_reversal_short):
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1w_donchian_weekly_trend_filter_v1"
timeframe = "6h"
leverage = 1.0