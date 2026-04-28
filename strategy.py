#!/usr/bin/env python3
# Hypothesis: 1d Donchian breakout with 1-week EMA trend filter and volume confirmation.
# In trending markets (price above/below weekly EMA), Donchian breakouts capture strong moves.
# Volume confirmation ensures breakouts have participation. Works in both bull and bear
# markets by using weekly EMA as dynamic trend filter. Targets 20-50 trades over 4 years.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need enough for EMA calculation
        return np.zeros(n)
    
    # Calculate weekly EMA (50-period)
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian channels (20-period) on daily data
    # Use rolling window with min_periods
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(high_max[i]) or 
            np.isnan(low_min[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below weekly EMA
        price_above_ema = close[i] > ema_50_1w_aligned[i]
        price_below_ema = close[i] < ema_50_1w_aligned[i]
        
        # Donchian breakout conditions
        breakout_up = high[i] > high_max[i-1]  # Current high above previous period's high max
        breakout_down = low[i] < low_min[i-1]  # Current low below previous period's low min
        
        # Entry conditions with volume confirmation
        long_entry = price_above_ema and breakout_up and volume_filter[i]
        short_entry = price_below_ema and breakout_down and volume_filter[i]
        
        # Exit conditions: when trend reverses or opposite Donchian breakout
        long_exit = (not price_above_ema) or breakout_down
        short_exit = (not price_below_ema) or breakout_up
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_DonchianBreakout_1wEMA50_TrendFilter_Volume"
timeframe = "1d"
leverage = 1.0