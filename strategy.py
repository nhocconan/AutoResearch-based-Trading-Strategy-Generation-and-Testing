#!/usr/bin/env python3
"""
1d_20Day_WeeklyTrend_Breakout
Hypothesis: Breakouts above/below 20-day high/low with volume confirmation and weekly EMA34 trend filter. This captures strong momentum moves while avoiding counter-trend trades. Weekly EMA ensures alignment with longer-term trend, reducing whipsaws in ranging markets. Designed for low trade frequency (<20/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 20-day high/low for breakout
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    highest_20 = high_series.rolling(window=20, min_periods=20).max().values
    lowest_20 = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume spike: >1.8x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    # Weekly EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 20  # Need 20-period lookback
    
    for i in range(start_idx, n):
        if (np.isnan(highest_20[i]) or 
            np.isnan(lowest_20[i]) or
            np.isnan(volume_spike[i]) or
            np.isnan(ema_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_1w_val = ema_1w_aligned[i]
        vol_spike = volume_spike[i]
        high_break = highest_20[i]
        low_break = lowest_20[i]
        
        if position == 0:
            # Long: break above 20-day high with volume and above weekly EMA
            if price > high_break and vol_spike and price > ema_1w_val:
                signals[i] = 0.25
                position = 1
            # Short: break below 20-day low with volume and below weekly EMA
            elif price < low_break and vol_spike and price < ema_1w_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price falls below 20-day low or below weekly EMA
            if price < low_break or price < ema_1w_val:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price rises above 20-day high or above weekly EMA
            if price > high_break or price > ema_1w_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_20Day_WeeklyTrend_Breakout"
timeframe = "1d"
leverage = 1.0