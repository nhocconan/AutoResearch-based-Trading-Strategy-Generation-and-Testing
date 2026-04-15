#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + Weekly Trend Filter
# Elder Ray calculates Bull Power (High - EMA13) and Bear Power (Low - EMA13).
# Long when Bull Power > 0 and rising, Bear Power < 0 and falling.
# Short when Bear Power < 0 and falling, Bull Power < 0 and rising.
# Weekly trend filter uses 200-period EMA on weekly timeframe: only long when price > weekly EMA200, only short when price < weekly EMA200.
# Works in bull markets (captures strength) and bear markets (captures weakness). Target: 50-150 total trades.
# Timeframe: 6h, HTF: 1w

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate EMA13 for Elder Ray
    close_series = pd.Series(close)
    ema13 = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Bull Power and Bear Power
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Calculate EMA200 on weekly for trend filter
    close_1w_series = pd.Series(close_1w)
    ema200_1w = close_1w_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align weekly EMA200 to 6h timeframe
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if np.isnan(ema200_1w_aligned[i]):
            continue
        
        # Long conditions: Bull Power positive and rising, price above weekly EMA200
        bull_rising = bull_power[i] > bull_power[i-1]
        long_condition = (bull_power[i] > 0 and bull_rising and close[i] > ema200_1w_aligned[i])
        
        # Short conditions: Bear Power negative and falling, price below weekly EMA200
        bear_falling = bear_power[i] < bear_power[i-1]
        short_condition = (bear_power[i] < 0 and bear_falling and close[i] < ema200_1w_aligned[i])
        
        # Exit conditions: power signals reverse or price crosses weekly EMA200
        exit_long = (bull_power[i] <= 0 or not bull_rising or close[i] < ema200_1w_aligned[i])
        exit_short = (bear_power[i] >= 0 or not bear_falling or close[i] > ema200_1w_aligned[i])
        
        if long_condition and position <= 0:
            position = 1
            signals[i] = base_size
        elif short_condition and position >= 0:
            position = -1
            signals[i] = -base_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
    
    return signals

name = "6h_ElderRay_WeeklyTrend_Filter"
timeframe = "6h"
leverage = 1.0