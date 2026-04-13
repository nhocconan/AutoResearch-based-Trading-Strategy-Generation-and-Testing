#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Donchian(20) breakout with 1d weekly pivot filter
    # Long when: price breaks above Donchian(20) high AND weekly pivot shows bullish bias (price > weekly PP)
    # Short when: price breaks below Donchian(20) low AND weekly pivot shows bearish bias (price < weekly PP)
    # Exit when: price returns to Donchian(20) midpoint OR weekly pivot bias flips
    # Uses discrete sizing (0.25) targeting 50-150 trades over 4 years.
    # Weekly pivot from 1d data provides structural bias that works in both bull/bear markets.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate weekly pivot points from prior week (using last 5 trading days approximation)
    # Weekly high/low/close from prior 5-day period
    lookback_week = 5
    if len(close_1d) < lookback_week:
        weekly_pivot = np.full(len(close_1d), np.nan)
        weekly_r1 = np.full(len(close_1d), np.nan)
        weekly_s1 = np.full(len(close_1d), np.nan)
    else:
        # Rolling weekly OHLC
        weekly_high = pd.Series(high_1d).rolling(window=lookback_week, min_periods=lookback_week).max().values
        weekly_low = pd.Series(low_1d).rolling(window=lookback_week, min_periods=lookback_week).min().values
        weekly_close = pd.Series(close_1d).rolling(window=lookback_week, min_periods=lookback_week).last().values
        
        # Weekly pivot point: (H + L + C) / 3
        weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
        # Weekly R1: (2 * P) - L
        weekly_r1 = (2 * weekly_pivot) - weekly_low
        # Weekly S1: (2 * P) - H
        weekly_s1 = (2 * weekly_pivot) - weekly_high
    
    # Align weekly pivot levels to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1d, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1d, weekly_s1)
    
    # Calculate Donchian(20) on 6h
    lookback_donch = 20
    highest_high = pd.Series(high).rolling(window=lookback_donch, min_periods=lookback_donch).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_donch, min_periods=lookback_donch).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(weekly_pivot_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(donchian_mid[i])):
            signals[i] = 0.0
            continue
        
        # Donchian breakout conditions
        breakout_up = close[i] > highest_high[i]
        breakout_down = close[i] < lowest_low[i]
        
        # Weekly pivot bias
        bullish_bias = close[i] > weekly_pivot_aligned[i]
        bearish_bias = close[i] < weekly_pivot_aligned[i]
        
        # Exit conditions: return to midpoint or bias flip
        exit_long = (close[i] < donchian_mid[i]) or (position == 1 and not bullish_bias)
        exit_short = (close[i] > donchian_mid[i]) or (position == -1 and not bearish_bias)
        
        # Entry conditions
        long_entry = breakout_up and bullish_bias and position != 1
        short_entry = breakout_down and bearish_bias and position != -1
        
        # Execute signals
        if long_entry:
            position = 1
            signals[i] = position_size
        elif short_entry:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
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

name = "6h_1d_donchian_weekly_pivot_v1"
timeframe = "6h"
leverage = 1.0