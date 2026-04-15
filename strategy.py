#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Daily Range Breakout with Volume Confirmation and 1w Trend Filter
# Breakouts above previous day's high or below previous day's low are traded only when
# confirmed by volume and weekly trend (EMA50 > EMA200 on 1w = bullish trend).
# Works in bull markets (breakouts up) and bear markets (breakouts down).
# Target: 20-80 total trades over 4 years (5-20/year).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data for previous day's high/low
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Load 1w data for trend filter (EMA50 and EMA200)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Previous day's high and low (shifted by 1 to avoid look-ahead)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    
    # Align previous day's high/low to 1d timeframe (same timeframe, but need alignment for proper timing)
    prev_high_1d_aligned = align_htf_to_ltf(prices, df_1d, prev_high_1d)
    prev_low_1d_aligned = align_htf_to_ltf(prices, df_1d, prev_low_1d)
    
    # Calculate EMA50 and EMA200 on 1w
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align weekly EMAs to 1d timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(prev_high_1d_aligned[i]) or np.isnan(prev_low_1d_aligned[i]) or
            np.isnan(ema50_1w_aligned[i]) or np.isnan(ema200_1w_aligned[i])):
            continue
        
        # Determine trend: bullish if EMA50 > EMA200, bearish if EMA50 < EMA200
        is_bullish = ema50_1w_aligned[i] > ema200_1w_aligned[i]
        is_bearish = ema50_1w_aligned[i] < ema200_1w_aligned[i]
        
        # Long entry: price breaks above previous day's high + volume confirmation + bullish trend
        if (close[i] > prev_high_1d_aligned[i] and
            volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
            is_bullish and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price breaks below previous day's low + volume confirmation + bearish trend
        elif (close[i] < prev_low_1d_aligned[i] and
              volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
              is_bearish and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse breakout or trend reversal
        elif position == 1 and (close[i] < prev_low_1d_aligned[i] or not is_bullish):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > prev_high_1d_aligned[i] or not is_bearish):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "1d_DailyRangeBreakout_Volume_1wTrend"
timeframe = "1d"
leverage = 1.0