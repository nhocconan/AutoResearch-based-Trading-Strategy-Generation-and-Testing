#!/usr/bin/env python3
"""
1d Donchian Breakout + 1w Trend + Volume Confirmation
Hypothesis: Daily Donchian(20) breakouts capture trend continuation. Weekly EMA(21) 
filters for higher timeframe trend alignment. Volume > 1.5x average confirms 
institutional participation. Designed for 1d timeframe to limit trades and 
reduce fee drag while capturing major moves in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian_breakout_1w_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # 1w EMA(21) for trend filter
    ema_21 = df_1w['close'].ewm(span=21, adjust=False).mean().values
    ema_21_1d = align_htf_to_ltf(prices, df_1w, ema_21)
    
    # Daily Donchian(20) - using previous 20 days (excluding current)
    # Highest high of last 20 days
    high_series = pd.Series(high)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    # Lowest low of last 20 days
    low_series = pd.Series(low)
    donchian_low = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume filter (>1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema_21_1d[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian low or trend reverses
            if close[i] < donchian_low[i] or close[i] < ema_21_1d[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high or trend reverses
            if close[i] > donchian_high[i] or close[i] > ema_21_1d[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Breakout long with trend alignment
            if (close[i] > donchian_high[i] and 
                close[i] > ema_21_1d[i] and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Breakout short with trend alignment
            elif (close[i] < donchian_low[i] and 
                  close[i] < ema_21_1d[i] and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals