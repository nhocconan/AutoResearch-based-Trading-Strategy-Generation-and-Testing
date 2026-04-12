#!/usr/bin/env python3
"""
1d_1w_DonchianBreakout_With_TrendFilter_v1
Hypothesis: Use weekly trend filter with daily Donchian channel breakouts for medium-term trend following.
Enter long when price breaks above 20-day high AND weekly close > weekly SMA50 (uptrend).
Enter short when price breaks below 20-day low AND weekly close < weekly SMA50 (downtrend).
Exit when price reverses to 10-day opposite channel or trend changes.
Designed to capture major trends in both bull and bear markets with low trade frequency.
Target: 20-60 total trades over 4 years (5-15/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_DonchianBreakout_With_TrendFilter_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === WEEKLY TREND FILTER ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    sma50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    sma50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma50_1w)
    trend_up = sma50_1w_aligned > close_1w  # weekly close above SMA50
    trend_down = sma50_1w_aligned < close_1w  # weekly close below SMA50
    
    # === DAILY DONCHIAN CHANNELS ===
    # 20-day high/low for breakout
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 10-day high/low for exit (faster reversal)
    high_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values
    low_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # start after warmup
        # Skip if not ready
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(high_10[i]) or np.isnan(low_10[i]) or
            np.isnan(sma50_1w_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Get current weekly trend
        week_trend_up = trend_up[i]
        week_trend_down = trend_down[i]
        
        # Long: break above 20-day high in weekly uptrend
        long_signal = (week_trend_up and 
                      close[i] > high_20[i])
        
        # Short: break below 20-day low in weekly downtrend
        short_signal = (week_trend_down and 
                       close[i] < low_20[i])
        
        # Exit: reverse to 10-day opposite channel or trend change
        exit_long = (position == 1 and 
                    (close[i] < low_10[i] or not week_trend_up))
        exit_short = (position == -1 and 
                     (close[i] > high_10[i] or not week_trend_down))
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals