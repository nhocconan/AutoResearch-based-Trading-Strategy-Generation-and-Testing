#!/usr/bin/env python3
"""
1d Donchian Breakout + 1w Trend + Volume Confirmation
Hypothesis: Daily Donchian(20) breakouts provide high-probability directional moves.
Weekly EMA(21) trend filter ensures alignment with higher timeframe momentum.
Volume > 1.5x 24-period average confirms institutional participation.
Designed for 1d timeframe to target 30-100 trades over 4 years with low frequency
and high signal quality in both bull and bear markets.
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
    
    # 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly EMA(21) for trend filter
    ema_21_1w = df_1w['close'].ewm(span=21, adjust=False).mean().values
    ema_21_1w_daily = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Daily Donchian channels (20-period)
    # Highest high and lowest low over past 20 days
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume filter (>1.5x 24-period average)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema_21_1w_daily[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian low or trend reverses
            if close[i] < donchian_low[i] or close[i] < ema_21_1w_daily[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high or trend reverses
            if close[i] > donchian_high[i] or close[i] > ema_21_1w_daily[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Breakout long with trend alignment
            if (close[i] >= donchian_high[i] and 
                close[i] > ema_21_1w_daily[i] and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Breakout short with trend alignment
            elif (close[i] <= donchian_low[i] and 
                  close[i] < ema_21_1w_daily[i] and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals