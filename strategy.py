#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour timeframe with 1-day Donchian channel breakout and weekly EMA filter
# Long when price breaks above 1-day Donchian high AND weekly close above 20-week EMA
# Short when price breaks below 1-day Donchian low AND weekly close below 20-week EMA
# Uses weekly trend filter to avoid counter-trend trades. Donchian provides clear breakout signals.
# Target: 100-200 total trades over 4 years (25-50/year) to stay within optimal range for 12h timeframe.

name = "12h_donchian20_1d_1w_ema_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian Channel (20-period) on daily timeframe
    df_1d = get_htf_data(prices, '1d')
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    
    daily_high_series = pd.Series(daily_high)
    daily_low_series = pd.Series(daily_low)
    
    donchian_high = daily_high_series.rolling(window=20, min_periods=20).max()
    donchian_low = daily_low_series.rolling(window=20, min_periods=20).min()
    
    # Align daily Donchian levels to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high.values)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low.values)
    
    # Weekly trend filter: EMA(20) on weekly close
    df_1w = get_htf_data(prices, '1w')
    weekly_close = df_1w['close'].values
    
    # Calculate 20-period EMA on weekly close
    weekly_close_series = pd.Series(weekly_close)
    weekly_ema = weekly_close_series.ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # Align weekly EMA to 12h timeframe
    weekly_ema_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not available
        if np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or np.isnan(weekly_ema_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: reverse position or trend change
        if position == 1:  # long position
            # Exit: price breaks below daily Donchian low or weekly trend turns bearish
            if (close[i] <= donchian_low_aligned[i] or 
                weekly_close[i] < weekly_ema_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above daily Donchian high or weekly trend turns bullish
            if (close[i] >= donchian_high_aligned[i] or 
                weekly_close[i] > weekly_ema_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with weekly trend filter
            # Long: price breaks above daily Donchian high AND weekly close above weekly EMA
            if (close[i] > donchian_high_aligned[i] and 
                weekly_close[i] > weekly_ema_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below daily Donchian low AND weekly close below weekly EMA
            elif (close[i] < donchian_low_aligned[i] and 
                  weekly_close[i] < weekly_ema_aligned[i]):
                signals[i] = -0.25
                position = -1
    
    return signals