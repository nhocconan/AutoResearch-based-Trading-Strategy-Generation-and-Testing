#!/usr/bin/env python3
# Hypothesis: 12h timeframe with 1-day Williams %R (overbought/oversold) and 1-week EMA trend filter.
# In oversold conditions (Williams %R < -80) with bullish trend (price > weekly EMA20), go long.
# In overbought conditions (Williams %R > -20) with bearish trend (price < weekly EMA20), go short.
# Uses Williams %R for mean-reversion signals and weekly EMA for trend filter to avoid counter-trend trades.
# Target: 50-150 total trades over 4 years (12-37/year) with size 0.25.

name = "12h_WilliamsR_WeeklyEMA_Filter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1-day Williams %R (14-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high']
    low_1d = df_1d['low']
    close_1d = df_1d['close']
    
    highest_high = high_1d.rolling(window=14, min_periods=14).max()
    lowest_low = low_1d.rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    williams_r = williams_r.replace(0, np.nan)  # Avoid division by zero
    
    williams_r_values = williams_r.values
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r_values)
    
    # Oversold/overbought thresholds
    oversold = williams_r_aligned < -80
    overbought = williams_r_aligned > -20
    
    # Calculate 1-week EMA (20-period) for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close']
    ema_20 = close_1w.ewm(span=20, adjust=False, min_periods=20).mean()
    ema_20_values = ema_20.values
    ema_20_aligned = align_htf_to_ltf(prices, df_1w, ema_20_values)
    
    # Trend conditions: bullish if price > weekly EMA20, bearish if price < weekly EMA20
    price_above_weekly_ema = close > ema_20_aligned
    price_below_weekly_ema = close < ema_20_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or
            np.isnan(ema_20_aligned[i]) or
            np.isnan(price_above_weekly_ema[i]) or np.isnan(price_below_weekly_ema[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: oversold (Williams %R < -80) + bullish trend (price > weekly EMA20)
            if oversold[i] and price_above_weekly_ema[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: overbought (Williams %R > -20) + bearish trend (price < weekly EMA20)
            elif overbought[i] and price_below_weekly_ema[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: overbought condition OR trend turns bearish
            if overbought[i] or (not price_above_weekly_ema[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: oversold condition OR trend turns bullish
            if oversold[i] or (not price_below_weekly_ema[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals