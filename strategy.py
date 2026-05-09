#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d 20-day breakout with 1w EMA trend filter and volume confirmation
# Daily breakout captures medium-term trends, 1w EMA filters for strong trend direction,
# and volume confirmation ensures institutional participation. Works in bull/bear markets
# by requiring trend alignment. Target: 30-100 trades over 4 years.
name = "1d_20DayBreakout_1wEMA_Trend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 20-day Donchian channels (highest high/lowest low of last 20 days)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    highest_high_20d = high_series.rolling(window=20, min_periods=20).max().values
    lowest_low_20d = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current volume > 1.5x 20-day average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for EMA and Donchian calculations
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d[i]) or np.isnan(highest_high_20d[i]) or 
            np.isnan(lowest_low_20d[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Breakout conditions
        long_breakout = close[i] > highest_high_20d[i-1]  # Break above 20-day high
        short_breakout = close[i] < lowest_low_20d[i-1]  # Break below 20-day low
        
        trend_up = close[i] > ema_50_1d[i]
        trend_down = close[i] < ema_50_1d[i]
        
        if position == 0:
            # Long: bullish breakout + uptrend + volume confirmation
            if long_breakout and trend_up and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish breakout + downtrend + volume confirmation
            elif short_breakout and trend_down and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bearish breakout below 20-day low or trend reversal
            if close[i] < lowest_low_20d[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish breakout above 20-day high or trend reversal
            if close[i] > highest_high_20d[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals