#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian Breakout with Weekly Trend Filter
# Uses 1-day Donchian(20) breakouts for entry, filtered by 1-week EMA50 trend.
# Only takes long when price breaks above Donchian upper band and weekly trend is up,
# short when price breaks below Donchian lower band and weekly trend is down.
# Designed to capture strong trends with minimal whipsaw by requiring
# weekly trend alignment. Target: 30-100 total trades over 4 years.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d and 1w data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_up = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align indicators to 1d timeframe
    donchian_up_aligned = align_htf_to_ltf(prices, df_1d, donchian_up)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50  # for Donchian and EMA calculations
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_up_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long setup: price breaks above Donchian upper band with weekly uptrend
            if (price > donchian_up_aligned[i] and    # Breakout above upper band
                price > ema_50_1w_aligned[i]):        # Above weekly EMA50 (uptrend)
                position = 1
                signals[i] = position_size
            # Short setup: price breaks below Donchian lower band with weekly downtrend
            elif (price < donchian_low_aligned[i] and  # Breakdown below lower band
                  price < ema_50_1w_aligned[i]):       # Below weekly EMA50 (downtrend)
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below Donchian lower band
            if price < donchian_low_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above Donchian upper band
            if price > donchian_up_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_Donchian_Breakout_1wEMA_Trend_Filter"
timeframe = "1d"
leverage = 1.0