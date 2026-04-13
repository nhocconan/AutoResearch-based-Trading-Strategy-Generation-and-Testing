#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with 1d trend filter and Williams %R mean reversion
# Strategy: In strong trends (price above/below 1d EMA50), look for mean reversion to Williams %R oversold/overbought levels
# This adapts to both bull/bear markets by using trend-aware mean reversion with momentum oscillator
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag
# Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for trend and Williams %R
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1d Williams %R (14 period)
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14 + 1e-10)
    
    # Align 1d indicators to 12h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(williams_r_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Williams %R levels
        oversold = williams_r_aligned[i] <= -80  # Oversold condition
        overbought = williams_r_aligned[i] >= -20  # Overbought condition
        
        # Trend direction from 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Entry logic: trend-following mean reversion
        long_entry = False
        short_entry = False
        
        if uptrend and oversold:
            # In uptrend, look for oversold bounces
            long_entry = True
        elif downtrend and overbought:
            # In downtrend, look for overbought pullbacks
            short_entry = True
        
        # Exit conditions: opposite signal or Williams %R reverts to middle
        exit_long = position == 1 and (williams_r_aligned[i] >= -50 or short_entry)
        exit_short = position == -1 and (williams_r_aligned[i] <= -50 or long_entry)
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
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

name = "12h_1d_williamsr_trend_mean_reversion_v1"
timeframe = "12h"
leverage = 1.0