#!/usr/bin/env python3
"""
1d_KAMA_Trend_With_Weekly_Filter
Hypothesis: On daily timeframe, use Kaufman's Adaptive Moving Average (KAMA) for trend direction. 
Filter with weekly trend (EMA13 > EMA34) to avoid counter-trend trades. Enter long when price 
crosses above KAMA with volume confirmation, short when price crosses below KAMA with volume confirmation. 
Exit on opposite KAMA cross. Weekly trend filter reduces whipsaws during ranging markets, 
while KAMA adapts to volatility. Designed for low trade frequency (~10-20/year) to minimize fee decay.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 34:
        return np.zeros(n)
    
    # Calculate weekly 13 and 34 EMA for trend filter
    close_weekly = df_weekly['close'].values
    ema13_weekly = pd.Series(close_weekly).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema34_weekly = pd.Series(close_weekly).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align weekly EMAs to daily timeframe
    ema13_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema13_weekly)
    ema34_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema34_weekly)
    
    # Weekly trend: bullish when EMA13 > EMA34
    weekly_uptrend = ema13_weekly_aligned > ema34_weekly_aligned
    weekly_downtrend = ema13_weekly_aligned < ema34_weekly_aligned
    
    # KAMA calculation (adaptive moving average)
    # Efficiency Ratio (ER) = |change| / sum(|changes|)
    change = np.abs(np.diff(close, prepend=close[0]))
    abs_change = np.abs(np.diff(close, prepend=close[0]))
    
    # Use 10-period for fast EMA, 30-period for slow EMA
    fast_span = 2
    slow_span = 30
    
    # Calculate ER over 10 periods
    er = np.zeros(n)
    for i in range(10, n):
        if i >= 10:
            direction = np.abs(close[i] - close[i-10])
            volatility = np.sum(np.abs(np.diff(close[i-9:i+1])))
            if volatility > 0:
                er[i] = direction / volatility
            else:
                er[i] = 0
    
    # Smoothing constants
    sc = (er * (2/(fast_span+1) - 2/(slow_span+1)) + 2/(slow_span+1)) ** 2
    
    # Calculate KAMA
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Volume confirmation: current volume > 1.5x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_surge = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema13_weekly_aligned[i]) or np.isnan(ema34_weekly_aligned[i]) or
            np.isnan(kama[i]) or np.isnan(volume_surge[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions with weekly trend alignment and volume confirmation
        long_entry = close[i] > kama[i] and close[i-1] <= kama[i-1] and weekly_uptrend[i] and volume_surge[i]
        short_entry = close[i] < kama[i] and close[i-1] >= kama[i-1] and weekly_downtrend[i] and volume_surge[i]
        
        # Exit on opposite KAMA cross
        long_exit = close[i] < kama[i] and close[i-1] >= kama[i-1]
        short_exit = close[i] > kama[i] and close[i-1] <= kama[i-1]
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_KAMA_Trend_With_Weekly_Filter"
timeframe = "1d"
leverage = 1.0