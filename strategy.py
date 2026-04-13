#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter and volume confirmation
    # Long when: price breaks above Donchian(20) high AND weekly pivot shows bullish bias AND volume > 2x 20-bar avg volume
    # Short when: price breaks below Donchian(20) low AND weekly pivot shows bearish bias AND volume > 2x 20-bar avg volume
    # Exit when: price crosses Donchian(20) midpoint
    # Uses discrete sizing (0.25) targeting 75-150 trades over 4 years.
    # Weekly pivot filter prevents counter-trend trades in ranging markets.
    # High volume threshold (2x) reduces false breakouts in choppy markets.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week's OHLC)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    weekly_open = df_1w['open'].values
    
    # Pivot point = (H + L + C) / 3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3
    # Resistance levels
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_r2 = weekly_pivot + (weekly_high - weekly_low)
    weekly_r3 = weekly_high + 2 * (weekly_pivot - weekly_low)
    # Support levels
    weekly_s1 = 2 * weekly_pivot - weekly_high
    weekly_s2 = weekly_pivot - (weekly_high - weekly_low)
    weekly_s3 = weekly_low - 2 * (weekly_high - weekly_pivot)
    
    # Determine weekly bias: bullish if close above pivot, bearish if below
    weekly_bullish = weekly_close > weekly_pivot
    weekly_bearish = weekly_close < weekly_pivot
    
    # Align weekly data to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish.astype(float))
    weekly_r3_aligned = align_htf_to_ltf(prices, df_1w, weekly_r3)
    weekly_s3_aligned = align_htf_to_ltf(prices, df_1w, weekly_s3)
    
    # Calculate Donchian(20) channels
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_mid = (highest_high + lowest_low) / 2
    
    # Calculate volume confirmation: volume > 2.0x 20-bar average volume (stricter filter)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (2.0 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(weekly_bullish_aligned[i]) or
            np.isnan(weekly_bearish_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Donchian breakout conditions
        breakout_up = close[i] > highest_high[i-1]  # Break above previous period high
        breakout_down = close[i] < lowest_low[i-1]  # Break below previous period low
        
        # Weekly pivot bias filter
        bullish_bias = weekly_bullish_aligned[i] > 0.5
        bearish_bias = weekly_bearish_aligned[i] > 0.5
        
        # Entry conditions with volume confirmation
        long_entry = breakout_up and bullish_bias and volume_confirmed[i] and position != 1
        short_entry = breakout_down and bearish_bias and volume_confirmed[i] and position != -1
        
        # Exit conditions
        exit_long = (position == 1 and close[i] < donchian_mid[i])
        exit_short = (position == -1 and close[i] > donchian_mid[i])
        
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

name = "6h_1w_donchian_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0