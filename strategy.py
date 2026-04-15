#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Weekly Range Breakout with Volume Confirmation and Trend Filter
# Uses previous week's high/low as support/resistance. Breakouts above weekly high or below weekly low
# are traded only when confirmed by volume (1.5x median) and weekly trend (close > weekly SMA50 for long, < for short).
# Works in bull markets (breakouts up) and bear markets (breakouts down). Target: 30-100 total trades.
# Timeframe: 1d, HTF: 1w

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data for previous week's high/low and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Previous week's high and low (shifted by 1 to avoid look-ahead)
    prev_high_1w = np.roll(high_1w, 1)
    prev_low_1w = np.roll(low_1w, 1)
    prev_high_1w[0] = np.nan  # First value has no previous week
    prev_low_1w[0] = np.nan
    
    # Align previous week's high/low to daily timeframe
    prev_high_1w_aligned = align_htf_to_ltf(prices, df_1w, prev_high_1w)
    prev_low_1w_aligned = align_htf_to_ltf(prices, df_1w, prev_low_1w)
    
    # Weekly trend: close > SMA50 for bullish, < SMA50 for bearish
    sma50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    weekly_bullish = close_1w > sma50_1w
    weekly_bearish = close_1w < sma50_1w
    
    # Align weekly trend to daily timeframe
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish.astype(float))
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(prev_high_1w_aligned[i]) or np.isnan(prev_low_1w_aligned[i]) or
            np.isnan(weekly_bullish_aligned[i]) or np.isnan(weekly_bearish_aligned[i])):
            continue
        
        # Long entry: price breaks above previous week's high + volume confirmation + weekly bullish trend
        if (close[i] > prev_high_1w_aligned[i] and
            volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
            weekly_bullish_aligned[i] > 0.5 and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price breaks below previous week's low + volume confirmation + weekly bearish trend
        elif (close[i] < prev_low_1w_aligned[i] and
              volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
              weekly_bearish_aligned[i] > 0.5 and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse breakout or trend change
        elif position == 1 and (close[i] < prev_low_1w_aligned[i] or weekly_bullish_aligned[i] <= 0.5):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > prev_high_1w_aligned[i] or weekly_bearish_aligned[i] <= 0.5):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "1d_1w_Range_Breakout_Volume_Trend"
timeframe = "1d"
leverage = 1.0