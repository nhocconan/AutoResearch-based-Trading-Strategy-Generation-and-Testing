#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day Donchian breakout with weekly EMA filter - trades only in direction of weekly trend
# Uses 20-day Donchian channels for breakouts and 20-week EMA for trend filter
# Designed to catch trends while avoiding counter-trend trades in ranging markets
# Target: 30-100 trades over 4 years by requiring both breakout and trend alignment

name = "1d_donchian20_1w_ema_trend_v2"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian Channel (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    
    donchian_high = high_series.rolling(window=20, min_periods=20).max()
    donchian_low = low_series.rolling(window=20, min_periods=20).min()
    
    # Weekly trend filter: EMA(20) on weekly close
    df_1w = get_htf_data(prices, '1w')
    weekly_close = df_1w['close'].values
    
    # Calculate 20-period EMA on weekly close
    weekly_close_series = pd.Series(weekly_close)
    weekly_ema = weekly_close_series.ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # Align weekly EMA to daily timeframe
    weekly_ema_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if weekly EMA data not available
        if np.isnan(weekly_ema_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: reverse position or trend change
        if position == 1:  # long position
            # Exit: price breaks below 20-day low or weekly close below EMA
            if (close[i] <= donchian_low[i] or 
                weekly_close[i] < weekly_ema_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above 20-day high or weekly close above EMA
            if (close[i] >= donchian_high[i] or 
                weekly_close[i] > weekly_ema_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with weekly trend filter
            # Long: price breaks above 20-day high AND weekly close above weekly EMA
            if (close[i] > donchian_high[i] and 
                weekly_close[i] > weekly_ema_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-day low AND weekly close below weekly EMA
            elif (close[i] < donchian_low[i] and 
                  weekly_close[i] < weekly_ema_aligned[i]):
                signals[i] = -0.25
                position = -1
    
    return signals