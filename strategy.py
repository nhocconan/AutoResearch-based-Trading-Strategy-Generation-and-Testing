#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 6h Donchian(20) breakout + 1d weekly pivot direction filter + volume confirmation
    # Long when: price breaks above 6h Donchian upper (20) AND weekly pivot bias is bullish AND volume > 1.5x 20-bar avg
    # Short when: price breaks below 6h Donchian lower (20) AND weekly pivot bias is bearish AND volume > 1.5x 20-bar avg
    # Exit when: price crosses 6h Donchian midpoint
    # Uses discrete sizing (0.25) targeting 75-150 total trades over 4 years.
    # Weekly pivot provides structural bias from higher timeframe; Donchian breakout gives entry; volume filters false breakouts.
    # Works in bull (breakouts with bullish bias) and bear (breakouts with bearish bias only).
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot points (using prior week's OHLC)
    # For each 1d bar, we need the prior week's (5 trading days) OHLC
    lookback = 5  # 5 trading days = 1 week
    week_high = pd.Series(high_1d).rolling(window=lookback, min_periods=lookback).max().values
    week_low = pd.Series(low_1d).rolling(window=lookback, min_periods=lookback).min().values
    week_close = pd.Series(close_1d).rolling(window=lookback, min_periods=lookback).last().values  # last close of the week
    week_open = pd.Series(close_1d).rolling(window=lookback, min_periods=lookback).first().values  # first open of the week
    
    # Weekly pivot: (week_high + week_low + week_close) / 3
    weekly_pivot = (week_high + week_low + week_close) / 3.0
    # Weekly R1: 2 * weekly_pivot - week_low
    weekly_r1 = 2 * weekly_pivot - week_low
    # Weekly S1: 2 * weekly_pivot - week_high
    weekly_s1 = 2 * weekly_pivot - week_high
    # Weekly bias: bullish if close > weekly_pivot, bearish if close < weekly_pivot
    weekly_bias_bullish = close_1d > weekly_pivot
    weekly_bias_bearish = close_1d < weekly_pivot
    
    # Align weekly bias to 6h timeframe
    weekly_bias_bullish_aligned = align_htf_to_ltf(prices, df_1d, weekly_bias_bullish.astype(float))
    weekly_bias_bearish_aligned = align_htf_to_ltf(prices, df_1d, weekly_bias_bearish.astype(float))
    
    # Calculate 6h Donchian channels (20-period)
    donchian_window = 20
    donchian_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Calculate volume confirmation: volume > 1.5x 20-bar average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(donchian_window, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(donchian_mid[i]) or
            np.isnan(weekly_bias_bullish_aligned[i]) or np.isnan(weekly_bias_bearish_aligned[i]) or
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Donchian breakout conditions (using current bar's close vs previous bar's levels)
        breakout_up = close[i] > donchian_high[i-1]  # break above previous Donchian high
        breakout_down = close[i] < donchian_low[i-1]  # break below previous Donchian low
        
        # Entry conditions with weekly pivot bias and volume confirmation
        long_entry = breakout_up and weekly_bias_bullish_aligned[i] == 1.0 and volume_confirmed[i] and position != 1
        short_entry = breakout_down and weekly_bias_bearish_aligned[i] == 1.0 and volume_confirmed[i] and position != -1
        
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

name = "6h_1d_donchian_weekly_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0