#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1D Price Action with 1W Trend Filter
# Uses daily close > weekly SMA50 for uptrend, < for downtrend.
# Enters on daily close crossing above/below 20-day high/low with volume confirmation.
# Designed to work in both bull and bear markets by aligning with weekly trend.
# Target: 20-60 trades over 4 years (5-15/year) with position size 0.25.
name = "1D_20DayBreakout_1WTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly SMA50 trend filter
    sma_50_1w = pd.Series(df_1w['close'].values).rolling(window=50, min_periods=50).mean().values
    sma_50_1d = align_htf_to_ltf(prices, df_1w, sma_50_1w)
    
    # Calculate daily 20-day high/low for breakout levels
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current volume > 1.5x 20-day average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for weekly SMA and daily calculations
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(sma_50_1d[i]) or np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend conditions
        trend_up = close[i] > sma_50_1d[i]
        trend_down = close[i] < sma_50_1d[i]
        
        # Breakout conditions
        breakout_up = close[i] > highest_20[i-1]  # Close above prior 20-day high
        breakout_down = close[i] < lowest_20[i-1]  # Close below prior 20-day low
        
        if position == 0:
            # Long: breakout above 20-day high + weekly uptrend + volume confirmation
            if breakout_up and trend_up and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: breakout below 20-day low + weekly downtrend + volume confirmation
            elif breakout_down and trend_down and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: close below 20-day low or trend reversal
            if close[i] < lowest_20[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: close above 20-day high or trend reversal
            if close[i] > highest_20[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals