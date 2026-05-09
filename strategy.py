#!/usr/bin/env python3
# Hypothesis: 12h breakout of daily high/low with volume confirmation and weekly trend filter
# Long when price breaks above daily high, volume > 1.5x 20-period average, and price above weekly EMA20
# Short when price breaks below daily low, volume > 1.5x 20-period average, and price below weekly EMA20
# Exit when price returns inside daily range (below daily high for long, above daily low for short)
# Position size: 0.25 (25% of capital) to balance return and drawdown
# Designed to work in trending markets via weekly trend filter and in ranging markets via mean reversion to daily range

name = "12h_DailyBreakout_Volume_WeeklyTrend"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly EMA20 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    weekly_ema20 = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    weekly_ema20_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema20)
    
    # Daily high/low for breakout levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_high_aligned = align_htf_to_ltf(prices, df_1d, daily_high)
    daily_low_aligned = align_htf_to_ltf(prices, df_1d, daily_low)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Need enough data for weekly EMA20 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(weekly_ema20_aligned[i]) or np.isnan(daily_high_aligned[i]) or 
            np.isnan(daily_low_aligned[i]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above daily high + volume spike + above weekly EMA20
            if (close[i] > daily_high_aligned[i] and 
                vol_spike[i] and 
                close[i] > weekly_ema20_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below daily low + volume spike + below weekly EMA20
            elif (close[i] < daily_low_aligned[i] and 
                  vol_spike[i] and 
                  close[i] < weekly_ema20_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns below daily high (mean reversion to daily range)
            if close[i] < daily_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns above daily low (mean reversion to daily range)
            if close[i] > daily_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals