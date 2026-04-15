#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Power with Weekly Trend Filter
# Uses Elder Ray (bull power = high - EMA13, bear power = low - EMA13) to measure bull/bear strength.
# Trades only when weekly trend agrees: bull power > 0 and weekly close > weekly EMA20 for longs,
# bear power < 0 and weekly close < weekly EMA20 for shorts.
# Works in bull markets (captures strength) and bear markets (captures weakness).
# Target: 50-150 total trades over 4 years = 12-37/year.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate EMA13 for Elder Ray
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    weekly_close = df_1w['close'].values
    
    # Weekly EMA20
    weekly_close_s = pd.Series(weekly_close)
    weekly_ema20 = weekly_close_s.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align weekly EMA20 to 6h timeframe
    weekly_ema20_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema20)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(weekly_ema20_aligned[i])):
            continue
        
        # Long entry: bull power positive AND weekly close above weekly EMA20
        if (bull_power[i] > 0 and
            weekly_close[-1] > weekly_ema20[-1] if len(weekly_close) > 0 else False and  # Current weekly close > weekly EMA20
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: bear power negative AND weekly close below weekly EMA20
        elif (bear_power[i] < 0 and
              weekly_close[-1] < weekly_ema20[-1] if len(weekly_close) > 0 else False and  # Current weekly close < weekly EMA20
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: opposite signal or power crosses zero
        elif position == 1 and (bull_power[i] < 0 or bear_power[i] > 0):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (bull_power[i] > 0 or bear_power[i] < 0):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "6h_ElderRay_WeeklyTrend_Filter"
timeframe = "6h"
leverage = 1.0